import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet3d.models.utils.grid_mask import GridMask
from mmdet.models.backbones.resnet import ResNet
from mmcv.cnn.bricks.conv_module import ConvModule
import math
from torch.distributions import Categorical
from pyquaternion import Quaternion
import torch.distributed as dist
# DISTRIBUTIONS
import mmcv
import sys
# from .. import builder
from einops import rearrange
import torch.nn as nn
from mmcv.cnn import build_plugin_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet3d.models.builder import build_voxel_encoder
import numpy as np
from plyfile import PlyData, PlyElement
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import os
import open3d as o3d
from ..backbones.diffusion.mv_diffuser import MultiViewDiffuser


def gamma_func(mode="cosine"):
    if mode == "linear":
        return lambda r: 1 - r
    elif mode == "cosine":
        return lambda r: np.cos(r * np.pi / 2)
    elif mode == "square":
        return lambda r: 1 - r**2
    elif mode == "cubic":
        return lambda r: 1 - r**3
    else:
        raise NotImplementedError
    
@DETECTORS.register_module()
class BEVDet_Render(CenterPoint):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self,
                 img_view_transformer,
                 #Diffusion
                 multi_view_diffuser_cfg,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 # BEV encoder & decoder using swin transformer
                 swin_bev_encoder=None,
                 swin_bev_decoder=None,
                 # VQ-VAE
                 vector_quantizer=None,
                 use_grid_mask=False,
                 use_vq=False,
                 **kwargs):
        super(BEVDet_Render, self).__init__(**kwargs)
        self.grid_mask = None if not use_grid_mask else \
            GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1,
                     prob=0.7)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        if img_bev_encoder_neck and img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = \
                builder.build_backbone(img_bev_encoder_backbone)
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        
        self.use_vq = use_vq

        if self.use_vq:
            # # BEV encoder & decoder using swin transformer
            self.swin_bev_encoder = build_transformer_layer_sequence(swin_bev_encoder)
            self.swin_bev_decoder = build_transformer_layer_sequence(swin_bev_decoder)

            # VQ-VAE
            self.vector_quantizer = builder.build_neck(vector_quantizer)
            self.pre_quant = nn.Sequential(nn.Linear(1024, 1024), nn.LayerNorm(1024))
            self.register_buffer("code_age", torch.zeros(self.vector_quantizer.n_e) * 10000)
            self.register_buffer("code_usage", torch.zeros(self.vector_quantizer.n_e))
            self.gamma = gamma_func("cosine")

            self.code_dict = {}
            for i in range(self.vector_quantizer.n_e):
                self.code_dict[i] = 0
            #### VQ-VAE end
        else:
            self.middle_layer = ConvModule(
                256,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv2d'))

        # Diffusion 
        self.multi_view_diffuser = MultiViewDiffuser(multi_view_diffuser_cfg)

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        if self.grid_mask is not None:
            imgs = self.grid_mask(imgs)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:7])
        x = self.bev_encoder(x)
        return [x], depth

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs) # img_feats: (B, C, H, W) (4, 256, 128, 128)
        

        losses = dict()

        if self.use_vq:
            # VQ stuff
            bev_feats = self.swin_bev_encoder(img_feats[0])
            feats = self.pre_quant(bev_feats)
            bev_quant, emb_loss, _ = self.vector_quantizer(feats, self.code_age, self.code_usage)
            bev_vqfeat = self.swin_bev_decoder(bev_quant)
            code_util = (self.code_age < self.vector_quantizer.dead_limit).sum() / self.code_age.numel()
            code_uniformity = self.code_usage.topk(10)[0].sum() / self.code_usage.sum()

            losses.update(
                {
                "loss_emb": sum(emb_loss) * 10,
                "code_util": code_util,
                "code_uniformity": code_uniformity,
                }
            )
            # VQ stuff end
        else:
            bev_vqfeat = self.middle_layer(img_feats[0])

        # ## Start diffusion
        camera_intrinsics = img_inputs[3] # (B, 6, 3, 3)
        post_rotations = img_inputs[4] # (B, 6, 3, 3)
        post_translations = img_inputs[5] # (B, 6, 3)
        post_translations_expanded = post_translations.unsqueeze(-1)  # Shape (B, 6, 3, 1)  
        transformation_matrices = torch.cat([post_rotations, post_translations_expanded], dim=-1)  # Shape (B, 6, 3, 4)  

        camera_param = torch.cat([
        camera_intrinsics,
        transformation_matrices
        ], dim=-1)  # Shape (B, 6, 3, 7)

        batch_size = img_inputs[0].shape[0]
        prompt_list = []
        for i in range(batch_size):
            prompt_list.append([{'location': "boston-seaport",
                'description': "It is a good day."}])
        diffuser_data_dict = dict(
            pixel_values=img_inputs[0],
            camera_param=camera_param,
            bev_vqfeat = bev_vqfeat,
            # hardcode prompt for now
            prompt = prompt_list
        )
        diffusion_results_loss = self.multi_view_diffuser(diffuser_data_dict)
        ## End diffusion


        # losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
        #                                     gt_labels_3d, img_metas,
        #                                     gt_bboxes_ignore)
        # losses.update(losses_pts)

        losses.update({
            "loss_diffuser": diffusion_results_loss['loss_diffuser'],
        })

        return losses

    # def forward_test(self,
    #                  points=None,
    #                  img_metas=None,
    #                  img_inputs=None,
    #                  **kwargs):
    #     """
    #     Args:
    #         points (list[torch.Tensor]): the outer list indicates test-time
    #             augmentations and inner torch.Tensor should have a shape NxC,
    #             which contains all points in the batch.
    #         img_metas (list[list[dict]]): the outer list indicates test-time
    #             augs (multiscale, flip, etc.) and the inner list indicates
    #             images in a batch
    #         img (list[torch.Tensor], optional): the outer
    #             list indicates test-time augmentations and inner
    #             torch.Tensor should have a shape NxCxHxW, which contains
    #             all images in the batch. Defaults to None.
    #     """
    #     for var, name in [(img_inputs, 'img_inputs'),
    #                       (img_metas, 'img_metas')]:
    #         if not isinstance(var, list):
    #             raise TypeError('{} must be a list, but got {}'.format(
    #                 name, type(var)))

    #     num_augs = len(img_inputs)
    #     if num_augs != len(img_metas):
    #         raise ValueError(
    #             'num of augmentations ({}) != num of image meta ({})'.format(
    #                 len(img_inputs), len(img_metas)))

    #     if not isinstance(img_inputs[0][0], list):
    #         img_inputs = [img_inputs] if img_inputs is None else img_inputs
    #         points = [points] if points is None else points
    #         return self.simple_test(points[0], img_metas[0], img_inputs[0],
    #                                 **kwargs)
    #     else:
    #         return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    # def aug_test(self, points, img_metas, img=None, rescale=False):
    #     """Test function without augmentaiton."""
    #     assert False

    # def simple_test(self,
    #                 points,
    #                 img_metas,
    #                 img=None,
    #                 rescale=False,
    #                 **kwargs):
    #     """Test function without augmentaiton."""
    #     img_feats, _, _ = self.extract_feat(
    #         points, img=img, img_metas=img_metas, **kwargs)
    #     bbox_list = [dict() for _ in range(len(img_metas))]
    #     bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
    #     for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
    #         result_dict['pts_bbox'] = pts_bbox
    #     return bbox_list

    # def forward_dummy(self,
    #                   points=None,
    #                   img_metas=None,
    #                   img_inputs=None,
    #                   **kwargs):
    #     img_feats, _, _ = self.extract_feat(
    #         points, img=img_inputs, img_metas=img_metas, **kwargs)
    #     assert self.with_pts_bbox
    #     outs = self.pts_bbox_head(img_feats)
    #     return outs
