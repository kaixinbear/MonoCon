model = dict(
    type='CenterNetMono3D',
    pretrained=True,
    backbone=dict(type='DLA', depth=34, norm_cfg=dict(type='BN')),
    neck=dict(
        type='DLAUp',
        in_channels_list=[64, 128, 256, 512],
        scales_list=(1, 2, 4, 8),
        start_level=2,
        norm_cfg=dict(type='BN')),
    bbox_head=dict(
        type='MonoConHead',
        in_channel=64,
        feat_channel=64,
        num_classes=3,
        num_alpha_bins=12,
        loss_center_heatmap=dict(
            type='CenterNetGaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_center2kpt_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_kpt_heatmap=dict(
            type='CenterNetGaussianFocalLoss', loss_weight=1.0),
        loss_kpt_heatmap_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_dim=dict(type='DimAwareL1Loss', loss_weight=1.0),
        loss_depth=dict(
            type='LaplacianAleatoricUncertaintyLoss', loss_weight=1.0),
        loss_alpha_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_alpha_reg=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=30, local_maximum_kernel=3, max_per_img=30, thresh=0.4))
dataset_type = 'KittiMonoDatasetMonoCon'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D', to_float32=True, color_type='color'),
    dict(
        type='LoadAnnotations3DMonoCon',
        with_bbox=True,
        with_2D_kpts=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomShiftMonoCon', shift_ratio=0.5, max_shift_px=32),
    dict(type='RandomFlipMonoCon', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car']),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers2d', 'depths',
            'gt_kpts_2d', 'gt_kpts_valid_mask'
        ],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'pad_shape', 'scale_factor', 'flip', 'cam_intrinsic',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'rect', 'Trv2c', 'P2',
                   'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                   'pcd_rotation', 'pts_filename', 'transformation_3d_flow',
                   'cam_intrinsic_p0'))
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAugMonoCon',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlipMonoCon'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Pedestrian', 'Cyclist', 'Car'],
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='KittiMonoDatasetMonoCon',
        data_root='data/kitti/',
        ann_file='data/kitti/kitti_infos_train_mono3d.coco.json',
        info_file='data/kitti/kitti_infos_train.pkl',
        img_prefix='data/kitti/',
        classes=['Pedestrian', 'Cyclist', 'Car'],
        pipeline=[
            dict(
                type='LoadImageFromFileMono3D',
                to_float32=True,
                color_type='color'),
            dict(
                type='LoadAnnotations3DMonoCon',
                with_bbox=True,
                with_2D_kpts=True,
                with_label=True,
                with_attr_label=False,
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox_depth=True),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(type='RandomShiftMonoCon', shift_ratio=0.5, max_shift_px=32),
            dict(type='RandomFlipMonoCon', flip_ratio_bev_horizontal=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Pedestrian', 'Cyclist', 'Car']),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
                    'gt_bboxes_3d', 'gt_labels_3d', 'centers2d', 'depths',
                    'gt_kpts_2d', 'gt_kpts_valid_mask'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'pad_shape', 'scale_factor', 'flip',
                           'cam_intrinsic', 'pcd_horizontal_flip',
                           'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                           'img_norm_cfg', 'rect', 'Trv2c', 'P2', 'pcd_trans',
                           'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                           'pts_filename', 'transformation_3d_flow',
                           'cam_intrinsic_p0'))
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        min_height=25,
        min_depth=2,
        max_depth=65,
        max_truncation=0.5,
        max_occlusion=2,
        box_type_3d='Camera'),
    val=dict(
        type='KittiMonoDatasetMonoCon',
        data_root='data/kitti/',
        ann_file='data/kitti/kitti_infos_val_mono3d.coco.json',
        info_file='data/kitti/kitti_infos_val.pkl',
        img_prefix='data/kitti/',
        classes=['Pedestrian', 'Cyclist', 'Car'],
        pipeline=[
            dict(type='LoadImageFromFileMono3D'),
            dict(
                type='MultiScaleFlipAugMonoCon',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='RandomFlipMonoCon'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Pedestrian', 'Cyclist', 'Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['img'])
                ])
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='Camera'),
    test=dict(
        type='KittiMonoDatasetMonoCon',
        data_root='data/kitti/',
        ann_file='data/kitti/kitti_infos_val_mono3d.coco.json',
        info_file='data/kitti/kitti_infos_val.pkl',
        img_prefix='data/kitti/',
        classes=['Pedestrian', 'Cyclist', 'Car'],
        pipeline=[
            dict(type='LoadImageFromFileMono3D'),
            dict(
                type='MultiScaleFlipAugMonoCon',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='RandomFlipMonoCon'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Pedestrian', 'Cyclist', 'Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['img'])
                ])
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='Camera'))
evaluation = dict(interval=5)
lr = 0.0001
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.95, 0.99),
    weight_decay=1e-05,
    paramwise_cfg=dict(
        bias_lr_mult=2.0, norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './tmp_work_dir'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
gpu_ids = [0]
