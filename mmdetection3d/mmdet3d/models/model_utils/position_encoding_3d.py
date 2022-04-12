import torch
import numpy as np
from mmdet3d.models.utils.gaussian_target_3D import bin_to_depth


def position_encoding_3D(H, W, D, scale_factor, camera_intrisc):
    xs = torch.linspace(0, W - 1, W, dtype=torch.float).type(torch.cuda.FloatTensor) * scale_factor
    ys = torch.linspace(0, H - 1, H, dtype=torch.float).type(torch.cuda.FloatTensor) * scale_factor
    zs_ = torch.linspace(0, D - 1, D, dtype=torch.float).type(torch.cuda.FloatTensor)
    zs = bin_to_depth(zs_)
    base_grid = torch.stack(torch.meshgrid(xs, ys, zs), dim=-1) 
    points = base_grid.view(-1, 3) # (u, v, D, 3)
    points3D = pts2Dto3D(points, camera_intrisc)
    points3D = points3D.view(W,H,D,-1).permute(1, 0, 2, 3).contiguous()        # (x, y, z, 3)
    points3D = normalize(points3D)
    # print(points3D, points3D.shape) # torch.Size([96, 312, 80, 4])
    H, W, D, _ = points3D.shape
    points3D = points3D.view(H, W, -1).permute(2, 0, 1).contiguous()
    return points3D


def pts2Dto3D(points, view):
    """
    Args:
        points (torch.Tensor): points in 2D images, [N, 3], \
            3 corresponds with x, y in the image and D.
        view (np.ndarray): camera instrinsic, [3, 3]

    Returns:
        torch.Tensor: points in 3D space. [N, 4], \
            4 corresponds with x, y, z, score in 3D space.
    """
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[1] == 3

    points2D = points[:, :2]
    depth = points[:, 2].view(-1, 1)
    unnorm_points2D = torch.cat([points2D * depth, depth], dim=1)

    viewpad = torch.eye(4, dtype=points2D.dtype, device=points2D.device)
    viewpad[:view.shape[0], :view.shape[1]] = points2D.new_tensor(view)
    inv_viewpad = torch.inverse(viewpad).transpose(0, 1)

    # Do operation in homogenous coordinates.
    nbr_points = unnorm_points2D.shape[0]
    homo_points2D = torch.cat(
        [unnorm_points2D,
            points2D.new_ones((nbr_points, 1))], dim=1)
    points3D = torch.mm(homo_points2D, inv_viewpad)
    return points3D

def normalize(point3d):
    '''
        Normalize to [0, 1]
        Args:
            points3d: (x, y, z, 4)
    '''
    x_min, x_max = torch.min(point3d[..., 0]), torch.max(point3d[..., 0])
    y_min, y_max = torch.min(point3d[..., 1]), torch.max(point3d[..., 1])
    z_min, z_max = torch.min(point3d[..., 2]), torch.max(point3d[..., 2])
    point3d[..., 0] = (point3d[..., 0] - x_min) / (x_max - x_min)
    point3d[..., 1] = (point3d[..., 1] - y_min) / (y_max - y_min)
    point3d[..., 2] = (point3d[..., 2] - z_min) / (z_max - z_min)
    return point3d

if __name__=='__main__':
    # 生成camera frustrum space
    # 注意特征图尺度是原图的1/4, 所以生成之后乘上4；
    W = 360
    H = 96
    D = 80
    scale_factor = 4
    camera_intrisc = np.array([[ 7.215377e+02,  0.000000e+00,  6.314407e+02, -4.485728e+01],
        [ 0.000000e+00,  7.215377e+02,  1.728540e+02,  2.163791e-01],
        [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  2.745884e-03],
        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
    position_encoding_3D(H, W, D, scale_factor, camera_intrisc)