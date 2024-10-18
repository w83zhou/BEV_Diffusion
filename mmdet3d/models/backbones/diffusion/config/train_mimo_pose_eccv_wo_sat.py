'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-02-29 21:45:32
Email: haimingzhang@link.cuhk.edu.cn
Description: Temporal depth consistency + render RGB alignment + multi-modal fusion.
'''
grad_max_norm = 35
print_freq = 10
max_epochs = 200
warmup_iters = 50

start_frame = 0

mid_frame = 5
end_frame = 11
plan_return_last = True
eval_length = 6

return_len_ = 11
return_len_train = 11
# load_from = 'out/vqvae/epoch_200.pth'
port = 25097
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


data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (200, 200),
    'src_size': (900, 1600),
    'render_size': (90, 160),  # (h, w)

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



data_path = 'data/nuscenes/'

train_dataset_config = dict(
    type='CustomNuScenesMultiModalDataset',
    need_render=True,
    input_dataset="gts",
    use_semantic=False,
    use_image=True,
    use_depth_consistency=True,
    render_size=data_config['render_size'],
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
    input_dataset="gts",
    need_render=True,
    use_semantic=False,  # we don't have the ground truth semantic in the validation set
    use_image=True,
    use_depth_consistency=True,
    render_size=data_config['render_size'],
    data_config=data_config,
    start_frame=start_frame,
    mid_frame=mid_frame,
    end_frame=end_frame,
    data_path = data_path,
    return_len = return_len_+1, 
    offset = 0,
    render_scale=[data_config['render_size'][0]/data_config['src_size'][0], 
                  data_config['render_size'][1]/data_config['src_size'][1]],
    imageset = 'data/nuscenes_infos_val_temporal_v3_scene.pkl', 
    test_mode=True
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
            weight=1.5,
            input_dict={
                'logits': 'logits',
                'labels': 'target_occs'}),
        dict(
            type='MSELoss',
            weight=10.0,
            input_dict={
                'input_imgs': 'img_pred',
                'target_imgs': 'target_imgs'}),
        dict(
            type='ImageAlignLoss',
            weight=5.0,
            input_dict={
                'input_imgs': 'render_image',
                'target_imgs': 'target_imgs'}),
        dict(
            type='PlanRegLossLidar',
            weight=0.1,
            loss_type='l2',
            num_modes=3,
            input_dict={
                'rel_pose': 'pose_pred',
                'metas': 'output_metas'})
    ]
)


loss_eval = dict(
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
    type='OccSATPredictor',
    debug=False,
    use_predictor=True,
    data_config=data_config,
    use_depth_consistency=True,
    depth_loss_weight=10.0,
    use_separate_head=True,
    opt=dict(
        avg_reprojection=False,
        disable_automasking=False,
        disparity_smoothness=0.001,
    ),
    model_cfg=dict(
        type='PlanMultiModalMIMOTransformer',
        patch_size=2,
        in_channel=expansion*_dim_*2*2,
        d_model=128,
        n_encoder_layers=6,
        n_decoder_layers=6,
        heads=8,
        history_len=mid_frame - start_frame,
        future_len=eval_length,
        encoder_cfg=dict(
            type='MMEncoder',
            in_channel=expansion*_dim_*2*2,
            img_in_channel=3*2*2,
            model_channel=128,
            N=6, ## number of layers
            heads=8,
            use_sensor_type_embed=True),
        img_decoder_cfg=dict(
            type='Decoder',
            channel=3*2*2,
            model_channel=128,
            N=6, ## number of layers
            heads=8),
        
        pose_encoder=dict(
            type='PoseEncoder',
            in_channels=5,
            out_channels=base_channel*2,
            num_layers=2,
            num_modes=3,
            num_fut_ts=1,
        ),
        pose_decoder=dict(
            type='PoseDecoder',
            in_channels=base_channel*2,
            num_layers=2,
            num_modes=3,
            num_fut_ts=1,
        ),
    ),
    render_head=dict(
        type='NeRFDecoderHead',
        mask_render=False,
        img_recon_head=True,
        semantic_head=False,
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
