'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-03-26 21:50:14
Email: haimingzhang@link.cuhk.edu.cn
Description: Validate the correctness of the pipeline under without flow setting.
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
            type='ReconLoss',
            weight=5.0,
            ignore_label=-100,
            use_weight=False,
            cls_weight=None,
            input_dict={
                'logits': 'coarse_logits',
                'labels': 'target_occs'}),
        dict(
            type='LovaszLoss',
            weight=0.5,
            input_dict={
                'logits': 'coarse_logits',
                'labels': 'target_occs'}),
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
    coarse_logits='coarse_logits'
)

base_channel = 64
_dim_ = 16
expansion = 8
n_e_ = 512
model = dict(
    type='XWorldPredictorWoFlow',
    use_predictor=True,
    model_cfg=dict(
        type='MIMOTransformer',
        patch_size=2,
        in_channel=expansion*_dim_*2*2,
        d_model=128,
        n_encoder_layers=6,
        n_decoder_layers=6,
        heads=8,
        history_len=mid_frame - start_frame,
        future_len=eval_length
    ),
    refine_decoder=dict(
        type='SimVP',
        shape_in=(6, 128, 200, 200),
        hid_S=16,
        groups=4,
    ),
)


shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]

unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = "./config/label_mapping/nuscenes-occ.yaml"
