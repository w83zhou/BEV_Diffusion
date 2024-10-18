'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-01-21 13:02:51
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
grad_max_norm = 35
print_freq = 10
max_epochs = 200
warmup_iters = 200
return_len_ = 10

multisteplr = False
multisteplr_config = dict(
    decay_t = [87 * 500],
    decay_rate = 0.1,
    warmup_t = warmup_iters,
    warmup_lr_init = 1e-6,
    t_in_epochs = False
)


optimizer = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=0.01,
    ),
)

data_path = 'data/nuscenes/'


train_dataset_config = dict(
    type='nuScenesSceneDatasetLidar',
    data_path = data_path,
    return_len = return_len_, 
    offset = 0,
    imageset = 'data/nuscenes_infos_train_temporal_v3_scene.pkl', 
)
    
val_dataset_config = dict(
    type='nuScenesSceneDatasetLidar',
    data_path = data_path,
    return_len = return_len_, 
    offset = 0,
    imageset = 'data/nuscenes_infos_val_temporal_v3_scene.pkl', 
)

train_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes',
    phase='train', 
)

val_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes',
    phase='val', 
)

train_loader = dict(
    batch_size = 1,
    shuffle = True,
    num_workers = 1,
)

val_loader = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)

    

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='ReconLoss',
            weight=10.0,
            ignore_label=-100,
            use_weight=False,
            cls_weight=None,
            input_dict={
                'logits': 'logits',
                'labels': 'inputs'}),
        dict(
            type='LovaszLoss',
            weight=1.0,
            input_dict={
                'logits': 'logits',
                'labels': 'inputs'}),
        dict(
            type='VQVAEEmbedLoss',
            weight=1.0),
        ])

loss_input_convertion = dict(
    logits='logits',
    embed_loss='embed_loss'
)


load_from = ''

_dim_ = 16
expansion = 8
base_channel = 64
n_e_ = 512
model = dict(
    type = 'VAERes3D',
    encoder_cfg=dict(
        type='Downsampler',
        in_channels = expansion, 
        downsample_steps = 2, 
    ), 
    decoder_cfg=dict(
        type='Upsampler',
        in_channels = 32, 
        upsampler_steps = 2, 
    ),
    num_classes=18,
    expansion=expansion, 
    vqvae_cfg=dict(
        type='VectorQuantizer',
        n_e = n_e_, 
        e_dim = base_channel * 2, 
        beta = 1., 
        z_channels=32,
        use_voxel=True))

shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]

unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = "./config/label_mapping/nuscenes-occ.yaml"