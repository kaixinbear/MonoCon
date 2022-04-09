from math import sqrt

import torch
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt



def bin_depths(depth_map, mode='LID', num_bins=80, depth_min=2.0, depth_max=46.8, target=True):
    """
    Converts depth map into bin indices
    Args:
        depth_map: (H, W), Depth Map
        mode: string, Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min: float, Minimum depth value
        depth_max: float, Maximum depth value
        num_bins: int, Number of depth bins
        target: bool, Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices: (H, W), Depth bin indices
    """
    float_flag = False
    if isinstance(depth_map, float) or isinstance(depth_map, int):
        depth_map = torch.tensor(depth_map, dtype=torch.float64)
        float_flag = True
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)
        if float_flag:
            indices = int(indices)
    return indices

def bin_to_depth(index, mode='LID', num_bins=80, depth_min=2.0, depth_max=46.8):
    '''
        reverse index to contigous depth
    '''
    if mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        depth = ((2 * (index + 0.5)) ** 2 - 1) * bin_size / 8 + depth_min
    return depth

def vis(heatmap):
    H, W, L = heatmap.shape[:3]
    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)
    z = np.arange(0, L, 1)

    x,y,z = np.meshgrid(x,y,z)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(x, z, y, c=heatmap[y, x, z], alpha=0.01)
    mask = heatmap[y, x, z] > 0.0
    # print(mask.sum())
    fg_x = x[mask]
    fg_y = y[mask]
    fg_z = z[mask]
    ax.scatter(fg_x, fg_z, fg_y,c=heatmap[fg_y, fg_x, fg_z], alpha=0.5)

def gaussian3D(radius_2d, depth_rel=6, sigma=1, dtype=torch.float32, device='cpu'):
    """Generate 3D gaussian kernel.
        Pay attention that radius on depth axis could not keep with radius on 2D.
        Set Depth radius as constant value at first.
        z_radius = 6 grid

    Args:
        radius_2d (int): Radius of 2D Bounding BBox gaussian kernel.
        depth_rel: Radius of depth gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1) * (2 * depth_rel + 1)`` shape.
    """
    radius = radius_2d
    x = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(-1, 1, 1)
    y = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, -1, 1)
    z = torch.arange(
        -depth_rel, depth_rel + 1, dtype=dtype, device=device).view(1, 1, -1)
    # z = torch.linspace(
    #     -depth_rel, depth_rel, 2 * radius, dtype=dtype, device=device).view(1, 1, -1)

    h = (-(x * x + y * y + z * z) / (2 * sigma * sigma)).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h

def gen_gaussian_3D_target(heatmap_3d, center, radius, depth_rel=6, k=1):
    """Generate 3D gaussian heatmap.

    Args:
        heatmap_3d (Tensor): Input 3D heatmap, [feat_H, feat_W, feat_D]. the gaussian kernel will cover on
            it and maintain the max value.
            set feat
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of 2D gaussian kernel.
        depth_rel (int): Absolute depth offset grid for z axis.
        k (int): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter = 2 * radius + 1
    gaussian_3d_kernel = gaussian3D(
        radius, depth_rel, sigma=diameter / 6, dtype=heatmap_3d.dtype, device=heatmap_3d.device)

    x, y, z = center # x, y refers to 2D BBox Center, z refers to depth channel 

    # contiguous depth to bin index
    z = bin_depths(z)

    height, width, length = heatmap_3d.shape[:3]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    forward, backward = min(z, depth_rel), min(length - z, depth_rel + 1)

    masked_heatmap = heatmap_3d[y - top:y + bottom, x - left:x + right, z - forward:z + backward]
    masked_gaussian = gaussian_3d_kernel[radius - top:radius + bottom,
                                      radius - left:radius + right,
                                      depth_rel - forward:depth_rel + backward]
    out_heatmap = heatmap_3d

    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right, z - forward:z+backward])

    return out_heatmap

def get_local_maximum(heat, kernel=3):
    """Extract local maximum pixel with given kernal.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool3d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def get_topk_from_3D_heatmap(scores, k=20):
    """Get top k positions from heatmap.

    Args:
        scores (Tensor): Target heatmap with shape
            [batch, num_classes, height, width].
        k (int): Target number. Default: 20.

    Returns:
        tuple[torch.Tensor]: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores (Tensor): Max scores of each topk keypoint.
        - topk_inds (Tensor): Indexes of each topk keypoint.
        - topk_clses (Tensor): Categories of each topk keypoint.
        - topk_ys (Tensor): Y-coord of each topk keypoint.
        - topk_xs (Tensor): X-coord of each topk keypoint.
    """
    batch, _, height, width, length = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (height * width * length)
    topk_inds = topk_inds % (height * width * length)
    topk_ys = topk_inds // (width * length)
    topk_inds = topk_inds % (width * length)
    topk_xs = topk_inds // length
    topk_zs = (topk_inds % length).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs, topk_zs

def gather_feat(feat, ind, mask=None):
    """
    Same as 2d.
    Gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.
        mask (Tensor | None): Mask of feature map. Default: None.

    Returns:
        feat (Tensor): Gathered feature.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def transpose_and_gather_feat(feat, ind):
    """Transpose and gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
            (B, C, H, W, D)
        ind (Tensor): Target coord index.

    Returns:
        feat (Tensor): Transposed and gathered feature.
            (B, N, C)
    """
    feat = feat.permute(0, 2, 3, 4, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(4))
    feat = gather_feat(feat, ind)
    return feat


if __name__=='__main__':
    # h = gaussian3D(radius_2d=10, depth_rel=3, sigma=1)
    # print(h)
    # heatmap_3d = torch.zeros([1, 3, 30, 70, 80])
    # # 35(cx) 对应 70(w), 12(cy) 对应 30(h)
    # gen_gaussian_3D_target(heatmap_3d[0, 0], center=[35, 12, 26], radius=4)


    # depth-level 的shape为(B, D+1, H, W)
    # 对所有前景depth_bin做conv得到Object level heatmap 3D pred
    # 生成 Object level heatmap 3D gt: (B, D, H, W)
    # 采用CenterNetGaussianFocalLoss

    # Per Object Heatmap3D Generator
    tmp_bins = 12
    depth = 10
    depth_bin = bin_depths(depth, mode='LID', num_bins=tmp_bins, depth_min=2.0, depth_max=46.8, target=True)
    print(depth_bin)
    heat_map_2d = torch.tensor([[0, 0.6, 0],
                               [0.6, 1, 0.6],
                               [0, 0.6, 0]])
    depth_dis = torch.zeros([tmp_bins])
    if depth_bin != tmp_bins:
        depth_dis[max(0, depth_bin-1):min(depth_bin+2, tmp_bins)] = 1   # 0.7
    heat_map_2d = heat_map_2d.unsqueeze(0)            # (1, ROI_H, ROI_W)
    depth_dis = depth_dis.unsqueeze(-1).unsqueeze(-1) # (D, 1, 1)
    print(depth_dis)
    heatmap_map_3d = heat_map_2d * depth_dis
    print(heatmap_map_3d)

    # Per Image Heatmap3D Generator
    heatmap3d = torch.zeros(D, H, W)
    for obj_id, gt_2D_Box in enumerate(gt_2D_Boxes):
        u1, v1, u2, v2 = gt_2D_Box
        heat_map_2d = heatmap2d[v1:v2, u1:u2]
        heat_map_3d = object_heat_map_3d(heat_map_2d, depths[obj_id])
        heatmap3d[:, v1:v2, u1:u2] = heat_map_3d