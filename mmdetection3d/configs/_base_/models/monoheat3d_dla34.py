model = dict(
    type='HeatMap_3D_Mono',
    pretrained=True,
    backbone=dict(
        type='DLA', depth=34, norm_cfg=dict(type='BN')),
    neck=dict(
        type='DLAUp',
        in_channels_list=[64, 128, 256, 512],
        scales_list=(1, 2, 4, 8),
        start_level=2,
        norm_cfg=dict(type='BN')),
    feat_2d_to_3d=dict(
        input_channel=64,
        output_channel=64,
        depth_bins=80,
        num_classes=3
    ),
    downsample_rate_on_4x=4,
    loss_2d=dict(
        num_classes=3,
        max_objs=30,
        loss_center_heatmap=dict(type='CenterNetGaussianFocalLoss', loss_weight=1.0, neg_loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0),    
    ),
    loss_depth_estimation=dict(
        weight=3.0,
        alpha=0.25,
        gamma=2.0,
        disc_cfg=dict(mode='LID', num_bins=80, depth_min=2.0, depth_max=46.8),
        fg_weight=13,
        bg_weight=1,
        downsample_factor=16
    ),
    heat3d_encoding=dict(input_channels=64, output_channel=32),
    bbox_head=dict(
        type='M3D_HeatMap_head',
        in_channel=32,
        feat_channel=32,
        num_classes=3,
        num_alpha_bins=12,
        loss_center_heatmap=dict(type='CenterNetGaussianFocalLoss', loss_weight=1.0, neg_loss_weight=0.01),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_center2d_to_3d_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_dim=dict(type='DimAwareL1Loss', loss_weight=1.0),
        loss_depth_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_alpha_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_alpha_reg=dict(type='L1Loss', loss_weight=1.0),
    ),
    train_cfg=None,
    test_cfg=dict(topk=30, local_maximum_kernel=3, max_per_img=30, 
    thresh=0.4, nms=dict(type='nms', iou_thr=0.5))
)
