_base_ = [
    '../_base_/models/monop2o_dla34_wo_p.py',
    '../_base_/datasets/kitti-mono3d-3class-monocon.py',
    '../_base_/schedules/cyclic_200e_monocon.py',
    '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=5)

workflow = [('train', 1)]
