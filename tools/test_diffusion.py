# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import mmdet
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from tools.train import load_hydra_cfg
from tqdm import tqdm
from mmdet3d.models.backbones.diffusion.misc.test_utils import prepare_pipe
from torchvision.transforms.functional import to_pil_image
from mmdet3d.models.backbones.diffusion.img_utils import concat_6_views, img_m11_to_01, img_concat_h, img_concat_v
from mmdet3d.datasets.pipelines.loading import mmlabDeNormalize
if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def convert_gt_image(gt_img):
    # bs, f, h, w, c = gt_img.shape
    # gt_img = gt_img.permute(0, 1, 4, 2, 3)
    # gt_img is of shape bs f c h w

    #ori_imgs = [to_pil_image(img_m11_to_01(img)) for img in gt_img[0]]
    ori_imgs = [to_pil_image(mmlabDeNormalize(img)) for img in gt_img[0]]
    return ori_imgs

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--save_vis', action='store_true', help='save vis results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def single_gpu_test(model,
                    data_loader,
                    cfg,
                    pipeline,
                    save_vis=True,
                    out_dir=None,):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        save_vis (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    hydra_cfg = cfg.model.multi_view_diffuser_cfg
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
            if hydra_cfg.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=data['img_inputs'][0].device).manual_seed(
                    hydra_cfg.seed
                )

            camera_intrinsics = data['img_inputs'][3] # (B, 6, 3, 3)
            post_rotations = data['img_inputs'][4] # (B, 6, 3, 3)
            post_translations = data['img_inputs'][5] # (B, 6, 3)
            post_translations_expanded = post_translations.unsqueeze(-1)  # Shape (B, 6, 3, 1)  
            transformation_matrices = torch.cat([post_rotations, post_translations_expanded], dim=-1)  # Shape (B, 6, 3, 4)  

            camera_param = torch.cat([
            camera_intrinsics,
            transformation_matrices
            ], dim=-1)  # Shape (B, 6, 3, 7)

            # Prompt
            batch_size = data['img_inputs'][0].shape[0]
            prompt_list = []
            for i in range(batch_size):
                prompt_list.append([{'location': "boston-seaport",
                    'description': "It is a good day."}])
            for _prompt in prompt_list:
                captions = []
                for example in _prompt:
                    caption = hydra_cfg.dataset.template.format(**example)
                    captions.append(caption)


            image = pipeline(
                image=result,
                camera_param=camera_param,
                prompt=captions,
                height=256, # bevdet takes 256 704
                width=704,
                generator=generator,
                **hydra_cfg.runner.pipeline_param,
            )

            if save_vis:
                # bs = 1
                image = image.images[0]
                image_seq = []
                for img in image:
                    image_seq.append(img)
                image = concat_6_views(image_seq)
                image.save(os.path.join(out_dir, f"gen_{i}.png"))
                gt_imgs = convert_gt_image(data['img_inputs'][0])
                gt_imgs = concat_6_views(gt_imgs)
                gt_imgs.save(os.path.join(out_dir, f"gt_{i}.png"))


        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.save_vis \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--save_vis" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # For diffusion
    hydra_cfg = load_hydra_cfg(
        config_path="../configs/bevdet_vq_render/configs", config_name="config",
        overrides=['+exp=224x400', "runner=8gpus"])
    cfg.model.multi_view_diffuser_cfg = hydra_cfg

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # checkpoint1 = torch.load(args.checkpoint)
    # for key in checkpoint1['state_dict'].keys():  
    #     print(key)  
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        pipeline, weight_dtype = prepare_pipe(hydra_cfg, model.multi_view_diffuser)
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        args.show_dir='./show_dir'
        args.save_vis=True
        if not os.path.exists(args.show_dir):  
             os.makedirs(args.show_dir)  
        outputs = single_gpu_test(model, data_loader, cfg, pipeline, args.save_vis, args.show_dir)
        #outputs = single_gpu_test(None, data_loader, cfg, None, args.save_vis, args.show_dir)
    else:
        assert distributed # not implemented yet
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

if __name__ == '__main__':
    main()
