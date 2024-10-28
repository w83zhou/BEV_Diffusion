point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset'
data_root = './data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0)),
    dict(
        type='BEVAug',
        bda_aug_conf=dict(
            rot_lim=(0.0, 0.0),
            scale_lim=(0.0, 0.0),
            flip_dx_ratio=0.0,
            flip_dy_ratio=0.0),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(type='Collect3D', keys=['img_inputs', 'magicdrive_img_inputs'])
]
test_pipeline = [
    dict(
        type='PrepareImageInputs',
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0)),
    dict(
        type='BEVAug',
        bda_aug_conf=dict(
            rot_lim=(0.0, 0.0),
            scale_lim=(0.0, 0.0),
            flip_dx_ratio=0.0,
            flip_dy_ratio=0.0),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        is_train=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='NuScenesDataset',
        data_root='./data/nuscenes/',
        ann_file='./data/nuscenes/bevdetv3-nuscenes_infos_train.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                is_train=True,
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0)),
            dict(
                type='BEVAug',
                bda_aug_conf=dict(
                    rot_lim=(0.0, 0.0),
                    scale_lim=(0.0, 0.0),
                    flip_dx_ratio=0.0,
                    flip_dy_ratio=0.0),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='Collect3D', keys=['img_inputs', 'magicdrive_img_inputs'])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=True,
        img_info_prototype='bevdet'),
    val=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='./data/nuscenes/bevdetv3-nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0)),
            dict(
                type='BEVAug',
                bda_aug_conf=dict(
                    rot_lim=(0.0, 0.0),
                    scale_lim=(0.0, 0.0),
                    flip_dx_ratio=0.0,
                    flip_dy_ratio=0.0),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                is_train=False),
            dict(type='Collect3D', keys=['img_inputs'])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='./data/nuscenes/bevdetv3-nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0)),
            dict(
                type='BEVAug',
                bda_aug_conf=dict(
                    rot_lim=(0.0, 0.0),
                    scale_lim=(0.0, 0.0),
                    flip_dx_ratio=0.0,
                    flip_dy_ratio=0.0),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                is_train=False),
            dict(type='Collect3D', keys=['img_inputs'])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './save_d/bevdet_r50_render'
load_from = './ckpts/bevdet-r50-bev-extractor.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(0.0, 0.0),
    rot=(0.0, 0.0),
    flip=False,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
grid_config = dict(
    x=[-51.2, 51.2, 0.8],
    y=[-51.2, 51.2, 0.8],
    z=[-5, 3, 8],
    depth=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Render',
    use_vq=False,
    bev_extractor=dict(
        type='BEV_Extractor',
        img_backbone=dict(
            pretrained='torchvision://resnet50',
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            with_cp=False,
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
            grid_config=dict(
                x=[-51.2, 51.2, 0.8],
                y=[-51.2, 51.2, 0.8],
                z=[-5, 3, 8],
                depth=[1.0, 60.0, 1.0]),
            input_size=(256, 704),
            in_channels=256,
            out_channels=64,
            downsample=16),
        img_bev_encoder_backbone=dict(
            type='CustomResNet', numC_input=64, num_channels=[128, 256, 512]),
        img_bev_encoder_neck=dict(
            type='FPN_LSS', in_channels=640, out_channels=256)))
bda_aug_conf = dict(
    rot_lim=(0.0, 0.0),
    scale_lim=(0.0, 0.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0)
share_data_config = dict(
    type='NuScenesDataset',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet')
test_data_config = dict(
    pipeline=[
        dict(
            type='PrepareImageInputs',
            data_config=dict(
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6,
                input_size=(256, 704),
                src_size=(900, 1600),
                resize=(0.0, 0.0),
                rot=(0.0, 0.0),
                flip=False,
                crop_h=(0.0, 0.0),
                resize_test=0.0)),
        dict(
            type='BEVAug',
            bda_aug_conf=dict(
                rot_lim=(0.0, 0.0),
                scale_lim=(0.0, 0.0),
                flip_dx_ratio=0.0,
                flip_dy_ratio=0.0),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            is_train=False),
        dict(type='Collect3D', keys=['img_inputs'])
    ],
    ann_file='./data/nuscenes/bevdetv3-nuscenes_infos_val.pkl',
    type='NuScenesDataset',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet')
key = 'test'
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=1e-07)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=12)
fp16 = dict(loss_scale='dynamic')
find_unused_parameters = True
gpu_ids = [0]
