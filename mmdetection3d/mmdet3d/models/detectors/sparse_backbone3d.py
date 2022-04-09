import torch.distributed as dist
from functools import partial

import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

class SparseBackbone3D(nn.Module):
    def __init__(self, input_channels, output_channel):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = post_act_block
        # self.conv1 = spconv.SparseSequential(
        #     spconv.SubMConv3d(input_channels, output_channel, kernel_size=3, stride=(1,2,2), padding=1, bias=False, indice_key='subm1'),
        #     norm_fn(output_channel),
        #     nn.ReLU(),
        # )

        self.conv1 = spconv.SparseSequential(
            block(input_channels, 32, 3, norm_fn=norm_fn, stride=(1, 1, 1), padding=(1, 1, 1), indice_key='spconv1', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(output_channel, 32, 3, norm_fn=norm_fn, stride=(1, 1, 1), padding=(1, 1, 1), indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

    def forward(self, input_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_indices = input_dict['voxel_features'], input_dict['voxel_indices']
        batch_size = input_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_indices.int(),
            spatial_shape=input_dict['sparse_shape'],
            batch_size=batch_size
        )

        x_conv1 = self.conv1(input_sp_tensor)
        # print("x_conv1", x_conv1.dense().shape)
        x_conv2 = self.conv2(x_conv1)
        # print("x_conv2", x_conv2.dense().shape)
        return x_conv2