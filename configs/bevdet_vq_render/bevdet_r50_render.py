_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':6,
    'input_size': (256, 704),
    # 'input_size': (64, 176),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (0.0, 0.0),
    'rot': (0.0, 0.0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64

model = dict(
    type='BEVDet_Render',
    use_vq=False,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=False, # Had problems using ckpt when using multiple GPUs
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        accelerate=True,
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    swin_bev_encoder=dict(
        type="VQEncoder",
        img_size=128,
        codebook_dim=1024,
    ),
    swin_bev_decoder=dict(
        type="VQDecoder",
        img_size=(128, 128),
        num_patches=256,
        codebook_dim=1024,
    ),
    vector_quantizer=dict(
        type="VectorQuantizer",
        n_e=1024,
        e_dim=1024,
        beta=0.25,
        cosine_similarity=False,
    ),
    
    # model training and testing settings
    # train_cfg=dict(
    #     pts=dict(
    #         point_cloud_range=point_cloud_range,
    #         grid_size=[1024, 1024, 40],
    #         voxel_size=voxel_size,
    #         out_size_factor=8,
    #         dense_reg=1,
    #         gaussian_overlap=0.1,
    #         max_objs=500,
    #         min_radius=2,
    #         code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    # test_cfg=dict(
    #     pts=dict(
    #         pc_range=point_cloud_range[:2],
    #         post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    #         max_per_img=500,
    #         max_pool_nms=False,
    #         min_radius=[4, 12, 10, 1, 0.85, 0.175],
    #         score_threshold=0.1,
    #         out_size_factor=8,
    #         voxel_size=voxel_size[:2],
    #         pre_max_size=1000,
    #         post_max_size=500,

    #         # Scale-NMS
    #         nms_type=['rotate'],
    #         nms_thr=[0.2],
    #         nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
    #                              1.1, 1.0, 1.0, 1.5, 3.5]]
    #     )
    # )
)

# Data
dataset_type = 'NuScenesDataset'
data_root = './data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(0.0, 0.0),
    scale_lim=(0.0, 0.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    # dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='DefaultFormatBundle3D', class_names=class_names),
    # dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
    dict(type='Collect3D', keys=['img_inputs'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    #
    # dict(type='LoadAnnotations'),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=5,
    #     file_client_args=file_client_args),
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type='DefaultFormatBundle3D',
    #             class_names=class_names,
    #             with_label=False),
    #         dict(type='Collect3D', keys=['points', 'img_inputs'])
    #     ])
    dict(type='Collect3D', keys=['img_inputs'])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl'
    )

data = dict(
    #samples_per_gpu=8,
    #workers_per_gpu=4,
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)

# Optimizer
# optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
# optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=200,
#     warmup_ratio=0.001,
#     step=[24,])
# runner = dict(type='EpochBasedRunner', max_epochs=24)

# From ultralidar
# optimizer = dict(
#     type="AdamW",
#     lr=8e-4,
#     betas=(0.9, 0.95),  # the momentum is change during training
#     paramwise_cfg=dict(
#         custom_keys={
#             "absolute_pos_embed": dict(decay_mult=0.0),
#             "relative_position_bias_table": dict(decay_mult=0.0),
#             "norm": dict(decay_mult=0.0),
#             "embedding": dict(decay_mult=0.0),
#             "img_backbone": dict(lr_mult=0.1, decay_mult=0.001),
#         }
#     ),
#     weight_decay=0.0001,
# )
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(policy="CosineAnnealing", warmup="linear", warmup_iters=500, warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)
runner = dict(type="EpochBasedRunner", max_epochs=12)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# fp16 = dict(loss_scale='dynamic')

load_from = None
validation_times = 4
find_unused_parameters = True

