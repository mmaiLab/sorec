_base_ = '../mm_grounding_dino/refcoco/grounding_dino_swin-t_finetune_8xb4_5e_refcoco.py'

# ===== dataset settings ======
data_root = 'data/'

train_pipeline_with_zoom = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode',
                   'zoom', 'bbox_history',))
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='SorecDataset',
        data_root=data_root,
        ann_file='sorec/trainSE.json',
        data_prefix=dict(img='sorec/trainSE/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        return_classes=True,
        pipeline=train_pipeline_with_zoom))

# ===== Val & Test dataset settings ======
test_pipeline_custom = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]
test_pipeline_in_predict = [
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]
eval_iou_thrs = (0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55, 0.6 ,0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95)

# Val
ann_file = 'sorec/val.json'
val_dataset_all_val = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='SODA-D/rawData/Images/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline_custom,
    backend_args=None)
val_evaluator_all_val = dict(
    type='SorecMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=eval_iou_thrs,
    topk=(1, 5, 10))

datasets = [val_dataset_all_val]
dataset_prefixes = ['sorec_val']
metrics = [val_evaluator_all_val]

val_dataloader = dict(
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)

# Test A
ann_file = 'sorec/testA.json'
test_dataset_sorec_testA = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='SODA-D/rawData/Images/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline_custom,
    backend_args=None)
test_evaluator_sorec_testA = dict(
    type='SorecMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=eval_iou_thrs,
    topk=(1, 5, 10))

# Test B
ann_file = 'sorec/testB.json'
test_dataset_sorec_testB = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='SODA-D/rawData/Images/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline_custom,
    backend_args=None)
test_evaluator_sorec_testB = dict(
    type='SorecMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=eval_iou_thrs,
    topk=(1, 5, 10))

datasets = [test_dataset_sorec_testA, test_dataset_sorec_testB]
dataset_prefixes = ['sorec_testA', 'sorec_testB']
metrics = [test_evaluator_sorec_testA, test_evaluator_sorec_testB]

test_dataloader = dict(
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)

# ===== model settings =====
common_pipeline_mean = [123.675, 116.28, 103.53]
common_pipeline_std = [58.395, 57.12, 57.375]
sorec_data_preprocessor = dict(
        type='SorecDetDataPreprocessor',
        mean=common_pipeline_mean,
        std=common_pipeline_std,
        bgr_to_rgb=True,
        pad_mask=False)

model = dict(
    type='PizaAdapterGroundingDINO',
    use_multistep_prediction=True,
    use_zoom_embed=True,
    use_bbox_history_encoder_embed=False,
    use_bbox_history_encoder=True,
    bbox_history_encoder_path=None,
    test_pipeline_in_predict=test_pipeline_in_predict,
    test_pipeline_mean=common_pipeline_mean,
    test_pipeline_std=common_pipeline_std,
    data_preprocessor = sorec_data_preprocessor,
    # Adapter setting
    adapter_type='adapter-plus',
    trainable_keys=[
        # Adapters
        "adapter1",
        "adapter2",
        # Additional Norms
        "norm",
        "LayerNorm",
        # Embeddings
        "zoom_embedding",
    ],
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
        }))

# ===== schedule settings =====
max_epochs = 5
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],
        gamma=0.1)
]

# ===== hook settings =====
wandb_backend = dict(type='WandbVisBackend',
                     init_kwargs={'project': 'sorec',
                                  'name': f'piza_adapter_trainS',
                                  'entity': 'wandb_account'})
vis_backends = [dict(type='LocalVisBackend'), wandb_backend]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5, save_best='auto'),
    logger=dict(type='LoggerHook', interval=200),
    visualization=dict(draw=False))
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
