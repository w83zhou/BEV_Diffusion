'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-02-21 23:35:03
Email: haimingzhang@link.cuhk.edu.cn
Description: Train the occworld with multi-modal inputs without predicting future images.
'''
grad_max_norm = 35
print_freq = 10
max_epochs = 200
warmup_iters = 50

start_frame = 0

mid_frame = 5
end_frame = 11
plan_return_last = True
eval_length = 6  # NOTE: here we hard code this length, since SimVP is designed for equal length input and output

return_len_ = 11
return_len_train = 11
# load_from = 'out/vqvae/epoch_200.pth'
port = 25092
revise_ckpt = 3
eval_every_epochs = 1
save_every_epochs = 1
multisteplr = False
multisteplr_config = dict(
    decay_t = [87 * 500],
    decay_rate = 0.1,
    warmup_t = warmup_iters,
    warmup_lr_init = 1e-6,
    t_in_epochs = False)

optimizer = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=0.01,
    ),
)

find_unused_parameters = True


data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'render_size': (90, 160),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

voxel_size = [16, 200, 200]
need_render = False


data_path = 'data/nuscenes/'

train_dataset_config = dict(
    type='CustomNuScenesMultiModalDataset',
    need_render=need_render,
    data_config=data_config,
    start_frame=start_frame,
    mid_frame=mid_frame,
    end_frame=end_frame,
    data_path = data_path,
    return_len = return_len_train+1, 
    offset = 0,
    render_scale=[data_config['render_size'][0]/data_config['src_size'][0], 
                  data_config['render_size'][1]/data_config['src_size'][1]],
    imageset = 'data/nuscenes_infos_train_temporal_v3_scene.pkl', 
)

val_dataset_config = dict(
    type='CustomNuScenesMultiModalDataset',
    need_render=need_render,
    data_config=data_config,
    start_frame=start_frame,
    mid_frame=mid_frame,
    end_frame=end_frame,
    data_path = data_path,
    return_len = return_len_+1, 
    offset = 0,
    imageset = 'data/nuscenes_infos_val_temporal_v3_scene.pkl', 
)

train_wrapper_config = dict(
    type='occdreamer_dataset_nuscenes',
    phase='train', 
)

val_wrapper_config = dict(
    type='occdreamer_dataset_nuscenes',
    phase='val', 
)

train_loader = dict(
    batch_size = 2,
    shuffle = True,
    num_workers = 2,
)
    
val_loader = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 2,
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
                'labels': 'target_occs'}),
        dict(
            type='LovaszLoss',
            weight=1.0,
            input_dict={
                'logits': 'logits',
                'labels': 'target_occs'}),
    ]
)


loss_input_convertion = dict(
    logits = 'logits',
)

base_channel = 64
_dim_ = 16
expansion = 8
n_e_ = 512
model = dict(
    type='MultiModalPredictor',
    debug=False,
    use_predictor=True,
    is_pred_imgs=False,
    model_cfg=dict(
        type='MIMOMultiModalTransformer',
        mmencoder_cfg=dict(
            type='MMEncoder',
            in_channel=expansion*_dim_*2*2,
            img_in_channel=3*2*2,
            model_channel=128,
            N=6, ## number of layers
            heads=8,
            use_sensor_type_embed=True),
        patch_size=2,
        in_channel=expansion*_dim_*2*2,
        img_in_channel=3*2*2,
        d_model=128,
        n_encoder_layers=6,
        n_decoder_layers=6,
        heads=8,
        history_len=mid_frame - start_frame,
        future_len=eval_length
    ),
    render_head=dict(
        type='NeRFDecoderHead',
        mask_render=False,
        img_recon_head=False,
        semantic_head=True,
        semantic_dim=17,
        real_size=grid_config['x'][:2] + grid_config['y'][:2] + grid_config['z'][:2],
        stepsize=grid_config['depth'][2],
        voxels_size=voxel_size,
        mode='bilinear',  # ['bilinear', 'nearest']
        render_type='density',  # ['prob', 'density']
        render_size=data_config['render_size'],
        depth_range=grid_config['depth'][:2],
        loss_nerf_weight=0.5,
        depth_loss_type='silog',  # ['silog', 'l1', 'rl1', 'sml1']
        variance_focus=0.85,  # only for silog loss
    ),
)


shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]

unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = "./config/label_mapping/nuscenes-occ.yaml"
