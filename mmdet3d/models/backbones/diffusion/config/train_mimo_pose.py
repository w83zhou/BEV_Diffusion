'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-01-25 23:16:18
Email: haimingzhang@link.cuhk.edu.cn
Description: Predict pose and occupancy together.
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
port = 25096
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



data_path = 'data/nuscenes/'

train_dataset_config = dict(
    type='CustomNuScenesDataset',
    start_frame=start_frame,
    mid_frame=mid_frame,
    end_frame=end_frame,
    data_path = data_path,
    return_len = return_len_train+1, 
    offset = 0,
    imageset = 'data/nuscenes_infos_train_temporal_v3_scene.pkl', 
)

val_dataset_config = dict(
    type='CustomNuScenesDataset',
    start_frame=start_frame,
    mid_frame=mid_frame,
    end_frame=end_frame,
    data_path = data_path,
    return_len = return_len_+1, 
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
        dict(
            type='PlanRegLossLidar',
            weight=0.1,
            loss_type='l2',
            num_modes=3,
            input_dict={
                'rel_pose': 'rel_pose',
                'metas': 'metas'})
    ]
)


loss_input_convertion = dict(
    logits='logits',
    rel_pose='pose_pred',
    metas='output_metas',
)

base_channel = 64
_dim_ = 16
expansion = 8
n_e_ = 512
model = dict(
    type='MIMOPredictor',
    use_predictor=True,
    model_cfg=dict(
        type='PlanMIMOTransformer',
        patch_size=2,
        in_channel=expansion*_dim_*2*2,
        d_model=128,
        n_encoder_layers=6,
        n_decoder_layers=6,
        heads=8,
        history_len=mid_frame - start_frame,
        future_len=eval_length,
        
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
    
)


unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = "./config/label_mapping/nuscenes-occ.yaml"
