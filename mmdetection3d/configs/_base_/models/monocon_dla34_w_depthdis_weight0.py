model = dict(
    type='CenterNetMono3D_W_DepthDis',
    pretrained=True,
    backbone=dict(
        type='DLA', depth=34, norm_cfg=dict(type='BN')),
    neck=dict(
        type='DLAUp',
        in_channels_list=[64, 128, 256, 512],
        scales_list=(1, 2, 4, 8),
        start_level=2,
        norm_cfg=dict(type='BN')),
    bbox_head=dict(
        type='MonoConHead_W_DepthDis',
        in_channel=64,
        feat_channel=64,
        num_classes=3,
        num_alpha_bins=12,
        loss_center_heatmap=dict(type='CenterNetGaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_center2kpt_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_kpt_heatmap=dict(type='CenterNetGaussianFocalLoss', loss_weight=1.0),
        loss_kpt_heatmap_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_dim=dict(type='DimAwareL1Loss', loss_weight=1.0),
        loss_depth=dict(type='LaplacianAleatoricUncertaintyLoss', loss_weight=1.0),
        loss_alpha_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_alpha_reg=dict(type='L1Loss', loss_weight=1.0),
        loss_depth_estimation=dict(
            weight=0.0,
            alpha=0.25,
            gamma=2.0,
            disc_cfg=dict(mode='LID', num_bins=80, depth_min=2.0, depth_max=46.8),
            fg_weight=13,
            bg_weight=1,
            downsample_factor=4
        ),
        use_AN=True,
        use_depth_estimate=True,
    ),
    train_cfg=None,
    test_cfg=dict(topk=30, local_maximum_kernel=3, max_per_img=30, thresh=0.4))
