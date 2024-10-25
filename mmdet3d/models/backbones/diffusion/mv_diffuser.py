import logging
import os
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from packaging import version
from diffusers import (
    ModelMixin,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from functools import partial
from einops import rearrange, repeat
import contextlib
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.utils.import_utils import is_xformers_available

from transformers.utils import ContextManagers
from .misc.common import (
    move_to,
    load_module,
    deepspeed_zero_init_disabled_context_manager,
    convert_outputs_to_fp16,
)
from ...builder import BACKBONES
from mmcv.runner import BaseModule, auto_fp16, wrap_fp16_model


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims
    dimensions.
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def _tokenize_captions(examples, template, tokenizer=None, is_train=True):
    captions = []
    for example in examples:
        caption = template.format(**example)
        captions.append(caption)
    captions.append("")
    if tokenizer is None:
        return None, captions

    # pad in the collate_fn function
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="do_not_pad",
        truncation=True,
    )

    input_ids = inputs.input_ids
    # pad to the longest of current batch (might differ between cards)
    padded_tokens = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    ).input_ids
    return padded_tokens, captions


class ControlnetUnetWrapper(ModelMixin):
    """As stated in https://github.com/huggingface/accelerate/issues/668, we
    should not use accumulate provided by accelerator, but create a wrapper to
    two modules.
    """

    def __init__(self, controlnet, unet, weight_dtype=torch.float32,
                 unet_in_fp16=True) -> None:
        super().__init__()
        self.controlnet = controlnet
        self.unet = unet
        self.weight_dtype = weight_dtype
        self.unet_in_fp16 = unet_in_fp16

    def forward(self, noisy_latents, timesteps, camera_param,
                encoder_hidden_states, encoder_hidden_states_uncond,
                controlnet_image, **kwargs):
        N_cam = noisy_latents.shape[1]
        kwargs = move_to(
            kwargs, self.weight_dtype, lambda x: x.dtype == torch.float32)

        # fmt: off
        down_block_res_samples, mid_block_res_sample, \
        encoder_hidden_states_with_cam = self.controlnet(
            noisy_latents,  # b, N_cam, 4, H/8, W/8
            timesteps,  # b
            camera_param=camera_param,  # b, N_cam, 189
            encoder_hidden_states=encoder_hidden_states,  # b, len, 768
            encoder_hidden_states_uncond=encoder_hidden_states_uncond,  # 1, len, 768
            controlnet_cond=controlnet_image,  # b, 26, 200, 200
            return_dict=False,
            **kwargs,
        )
        # fmt: on

        # starting from here, we use (B n) as batch_size
        noisy_latents = rearrange(noisy_latents, "b n ... -> (b n) ...")
        if timesteps.ndim == 1:
            timesteps = repeat(timesteps, "b -> (b n)", n=N_cam)

        # Predict the noise residual
        # NOTE: Since we fix most of the model, we cast the model to fp16 and
        # disable autocast to prevent it from falling back to fp32. Please
        # enable autocast on your customized/trainable modules.
        context = contextlib.nullcontext
        context_kwargs = {}
        if self.unet_in_fp16:
            context = torch.cuda.amp.autocast
            context_kwargs = {"enabled": False}
        with context(**context_kwargs):
            model_pred = self.unet(
                noisy_latents,  # b x n, 4, H/8, W/8
                timesteps.reshape(-1),  # b x n
                encoder_hidden_states=encoder_hidden_states_with_cam.to(
                    dtype=self.weight_dtype
                ),  # b x n, len + 1, 768
                # TODO: during training, some camera param are masked.
                down_block_additional_residuals=[
                    sample.to(dtype=self.weight_dtype)
                    for sample in down_block_res_samples
                ],  # all intermedite have four dims: b x n, c, h, w
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=self.weight_dtype
                ),  # b x n, 1280, h, w. we have 4 x 7 as mid_block_res
            ).sample

        model_pred = rearrange(model_pred, "(b n) ... -> b n ...", n=N_cam)
        return model_pred


class BaseDiffuser(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        # Load models and create wrapper for stable diffusion
        # workaround for ZeRO-3, see:
        # https://github.com/huggingface/diffusers/blob/3ebbaf7c96801271f9e6c21400033b6aa5ffcf29/examples/text_to_image/train_text_to_image.py#L571
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            self._init_fixed_models(cfg)
        self._init_trainable_models(cfg)

        # set model and xformers
        self._set_model_trainable_state()
        self._set_xformer_state()
        self._set_gradient_checkpointing()

        # param and placeholders
        self.weight_dtype = torch.float32
        self.overrode_max_train_steps = self.cfg.runner.max_train_steps is None
        self.num_update_steps_per_epoch = None  # based on train loader
        self.optimizer = None
        self.lr_scheduler = None

        # # validator
        # pipe_cls = load_module(cfg.model.pipe_module)
        # self.validator = BaseValidator(
        #     self.cfg,
        #     self.val_dataset,
        #     pipe_cls,
        #     pipe_param={
        #         "vae": self.vae,
        #         "text_encoder": self.text_encoder,
        #         "tokenizer": self.tokenizer,
        #     }
        # )

    def _set_xformer_state(self):
        # xformer
        if self.cfg.runner.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logging.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
                self.controlnet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly")

    def _set_gradient_checkpointing(self):
        if hasattr(self.cfg.runner.enable_unet_checkpointing, "__len__"):
            self.unet.enable_gradient_checkpointing(
                self.cfg.runner.enable_unet_checkpointing)
        elif self.cfg.runner.enable_unet_checkpointing:
            self.unet.enable_gradient_checkpointing()
        if self.cfg.runner.enable_controlnet_checkpointing:
            self.controlnet.enable_gradient_checkpointing()

@BACKBONES.register_module()
class MultiViewDiffuser(BaseDiffuser):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.is_init = True  # make this module not affected by the init_weights().
        self.unet_in_fp16 = True

        # self.fp16_enabled = False

    def _init_fixed_models(self, cfg):
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae")
        self.noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")

        # fmt: off
        unet = UNet2DConditionModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="unet")
        # fmt: on

        model_cls = load_module(cfg.model.unet_module)
        unet_param = OmegaConf.to_container(self.cfg.model.unet, resolve=True)
        self.unet = model_cls.from_unet_2d_condition(unet, **unet_param)
        unet_path = os.path.join('magicdrive-log/SDv1.5mv-rawbox_2023-09-07_18-39_224x400', 'unet')
        logging.info(f"Loading unet from {unet_path} with {self.unet}")
        self.unet = self.unet.from_pretrained(
            unet_path, torch_dtype=torch.float16)


    def _init_trainable_models(self, cfg):
        # fmt: off
        unet = UNet2DConditionModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="unet")
        # fmt: on

        # model_cls = load_module(cfg.model.unet_module)
        # unet_param = OmegaConf.to_container(self.cfg.model.unet, resolve=True)
        # self.unet = model_cls.from_unet_2d_condition(unet, **unet_param)

        model_cls = load_module(cfg.model.model_module)
        controlnet_param = OmegaConf.to_container(
            self.cfg.model.controlnet, resolve=True)
        self.controlnet = model_cls.from_unet(unet, **controlnet_param)

        # self.controlnet_unet = ControlnetUnetWrapper(self.controlnet, self.unet)

    def _set_model_trainable_state(self, train=True):
        # set trainable status
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.train(train)
        self.unet.requires_grad_(False)
        # for name, mod in self.unet.trainable_module.items():
        #     logging.debug(
        #         f"[MultiViewDiffuser] set {name} to requires_grad = True")
        #     mod.requires_grad_(train)

    def prepare_device(self):
        self.controlnet_unet = ControlnetUnetWrapper(self.controlnet, self.unet)

        self.weight_dtype = torch.float16
        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(dtype=self.weight_dtype)
        self.text_encoder.to(dtype=self.weight_dtype)
        self.unet.to(dtype=self.weight_dtype)
        # for name, mod in self.unet.trainable_module.items():
        #     logging.debug(f"[MultiviewRunner] set {name} to fp32")
        #     mod.to(dtype=torch.float32)
        #     mod._original_forward = mod.forward
        #     # autocast intermediate is necessary since others are fp16
        #     mod.forward = torch.cuda.amp.autocast(
        #         dtype=torch.float16)(mod.forward)
        #     # we ensure output is always fp16
        #     mod.forward = convert_outputs_to_fp16(mod.forward)

        self.controlnet_unet.weight_dtype = self.weight_dtype
        self.controlnet_unet.unet_in_fp16 = self.cfg.runner.unet_in_fp16

        # with torch.no_grad():
        #     self.accelerator.unwrap_model(self.controlnet).prepare(
        #         self.cfg,
        #         tokenizer=self.tokenizer,
        #         text_encoder=self.text_encoder
        #     )

    def tokenize_captions(self, batch):
        # 
        # prompt_list = batch["prompt"]
        # output_dict = []
        # for _prompt in prompt_list:
        #     input_ids_padded, captions = _tokenize_captions(
        #         _prompt,
        #         self.cfg.dataset.template,
        #         self.tokenizer,
        #         self.training)
        #     ret_dict = {}
        #     ret_dict["captions"] = captions[:-1]  # list of str
        #     # real captions in head; the last one is null caption
        #     # we omit "attention_mask": padded_tokens.attention_mask, seems useless
        #     ret_dict["input_ids"] = input_ids_padded[:-1]
        #     ret_dict["uncond_ids"] = input_ids_padded[-1:]
        #     output_dict.append(ret_dict)
        
        # output_dict = {k: [d[k] for d in output_dict] for k in output_dict[0]}
        prompt_list = batch["prompt"]
        output_dict = []
        for _prompt in prompt_list:
            input_ids_padded, captions = _tokenize_captions(
                    _prompt,
                    self.cfg.dataset.template,
                    self.tokenizer,
                    self.training)
            input_ids_padded = input_ids_padded.cuda()
            ret_dict = {}
            ret_dict["captions"] = captions[:-1]  # list of str
            # real captions in head; the last one is null caption
            # we omit "attention_mask": padded_tokens.attention_mask, seems useless
            
            ret_dict["input_ids"] = input_ids_padded[:-1]
            ret_dict["uncond_ids"] = input_ids_padded[-1:]
            output_dict.append(ret_dict)
        return output_dict

    # @auto_fp16()
    def forward(self, batch):
        self.vae.eval()
        self.text_encoder.eval()

        ret_dict = self.tokenize_captions(batch)
        batch.update(ret_dict[0]) ## NOTE: we only take the first one in the list 
            
        N_cam = batch["pixel_values"].shape[1]

        # Convert images to latent space
        pixel_values = rearrange(batch["pixel_values"], "b n c h w -> (b n) c h w")
        latents = self.vae.encode(
            pixel_values.to(
                dtype=self.weight_dtype
            )
        ).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam) # b x n x 4 x H/8 x W/8

        ## NOTE: we do not need the camera params in our settings
        # # embed camera params, in (B, 6, 3, 7), out (B, 6, 189)
        # # camera_emb = self._embed_camera(batch["camera_param"])
        camera_param = batch["camera_param"].to(self.weight_dtype)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        # make sure we use same noise for different views, only take the
        # first
        if self.cfg.model.train_with_same_noise:
             noise = repeat(noise[:, 0], "b ... -> b r ...", r=N_cam)

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        if self.cfg.model.train_with_same_t:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
        else:
            timesteps = torch.stack([torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            ) for _ in range(N_cam)], dim=1)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self._add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
        encoder_hidden_states_uncond = self.text_encoder(
            batch
            ["uncond_ids"])[0]

        controlnet_image = batch["bev_vqfeat"].to(
            dtype=self.weight_dtype)

        with torch.cuda.amp.autocast(enabled=True):
            model_pred = self.process(
                noisy_latents, timesteps, camera_param,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_uncond=encoder_hidden_states_uncond, 
                controlnet_image=controlnet_image,
            )

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        loss = F.mse_loss(
            model_pred.float(), target.float(), reduction='none')
        loss = loss.mean()

        if not loss.isfinite():
            raise RuntimeError('Your diffusion loss is NaN.')

        output_dict = {}
        output_dict['loss_diffuser'] = loss
        return output_dict
    
    def process(self, noisy_latents, timesteps, camera_param,
                encoder_hidden_states=None, 
                encoder_hidden_states_uncond=None,
                controlnet_image=None, **kwargs):
        N_cam = noisy_latents.shape[1]

        # fmt: off
        down_block_res_samples, mid_block_res_sample, \
        encoder_hidden_states_with_cam = self.controlnet(
            noisy_latents,  # b, N_cam, 4, H/8, W/8
            timesteps,  # b
            camera_param=camera_param,  # b, N_cam, 189
            encoder_hidden_states=encoder_hidden_states,  # b, len, 768
            encoder_hidden_states_uncond=encoder_hidden_states_uncond,  # 1, len, 768
            controlnet_cond=controlnet_image,  # b, 26, 200, 200
            bboxes_3d_data=None,
            return_dict=False,
        )
        # fmt: on

        # starting from here, we use (B n) as batch_size
        noisy_latents = rearrange(noisy_latents, "b n ... -> (b n) ...")
        if timesteps.ndim == 1:
            timesteps = repeat(timesteps, "b -> (b n)", n=N_cam)

        # Predict the noise residual
        # NOTE: Since we fix most of the model, we cast the model to fp16 and
        # disable autocast to prevent it from falling back to fp32. Please
        # enable autocast on your customized/trainable modules.
        context = contextlib.nullcontext
        context_kwargs = {}
        if self.unet_in_fp16:
            context = torch.cuda.amp.autocast
            context_kwargs = {"enabled": False}
        with context(**context_kwargs):
            model_pred = self.unet(
                noisy_latents,  # b x n, 4, H/8, W/8
                timesteps.reshape(-1),  # b x n
                encoder_hidden_states=encoder_hidden_states_with_cam.to(
                    dtype=self.weight_dtype
                ),  # b x n, len + 1, 768
                # TODO: during training, some camera param are masked.
                down_block_additional_residuals=[
                    sample.to(dtype=self.weight_dtype)
                    for sample in down_block_res_samples
                ],  # all intermedite have four dims: b x n, c, h, w
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=self.weight_dtype
                ),  # b x n, 1280, h, w. we have 4 x 7 as mid_block_res
            ).sample

        model_pred = rearrange(model_pred, "(b n) ... -> b n ...", n=N_cam)
        return model_pred
    
    def _add_noise(self, latents, noise, timesteps):
        if self.cfg.runner.noise_offset > 0.0:
            # noise offset in SDXL, see:
            # https://github.com/Stability-AI/generative-models/blob/45c443b316737a4ab6e40413d7794a7f5657c19f/sgm/modules/diffusionmodules/loss.py#L47
            # they did not apply on different channels. Don't know why.
            offset = self.cfg.runner.noise_offset * append_dims(
                torch.randn(latents.shape[:2], device=latents.device),
                latents.ndim
            ).type_as(latents)
            if self.cfg.runner.train_with_same_offset:
                offset = offset[:, :1]
            noise = noise + offset
        if timesteps.ndim == 2:
            B, N = latents.shape[:2]
            bc2b = partial(rearrange, pattern="b n ... -> (b n) ...")
            b2bc = partial(rearrange, pattern="(b n) ... -> b n ...", b=B)
        elif timesteps.ndim == 1:
            def bc2b(x): return x
            def b2bc(x): return x
        noisy_latents = self.noise_scheduler.add_noise(
            bc2b(latents), bc2b(noise), bc2b(timesteps)
        )
        noisy_latents = b2bc(noisy_latents)
        return noisy_latents