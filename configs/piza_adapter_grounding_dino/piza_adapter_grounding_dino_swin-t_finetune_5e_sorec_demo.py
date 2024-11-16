_base_ = '../mm_grounding_dino/refcoco/grounding_dino_swin-t_finetune_8xb4_5e_refcoco.py'

data_root = 'data/'

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


common_pipeline_mean = [123.675, 116.28, 103.53]
common_pipeline_std = [58.395, 57.12, 57.375]

model = dict(
    type='DemoPizaAdapterGroundingDINO',

    use_2step_prediction=True,
    use_multistep_prediction=True,
    use_zoom_embed=False,
    use_bbox_history=True,
    
    separate_bbox_history_module_path="",
    test_pipeline_in_predict=test_pipeline_in_predict,
    test_pipeline_mean=common_pipeline_mean,
    test_pipeline_std=common_pipeline_std,

    encoder=dict(
        layer_cfg=dict(adapter_bottle_neck_dim=512),
        text_layer_cfg=dict(adapter_bottle_neck_dim=512),
    ),

    adapter_type='adapter-plus',
    use_img_feat_adapter=True, 
    use_text_feat_adapter=True,
    trainable_keys=[
        "adapter1",
        "adapter2",
        "norm",
        "LayerNorm",
        "bbox_embedding",
        "zoom_embedding",
    ],
)

