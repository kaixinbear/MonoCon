import torch
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp
from torch import nn
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.hash import HashTable
import time

from mmdet.core import multi_apply
from mmdet.models.utils.gaussian_target import gaussian_radius
from mmdet3d.models.utils.gaussian_target_3D import gen_gaussian_3D_target, bin_depths, bin_to_depth, \
    get_local_maximum, get_topk_from_3D_heatmap, transpose_and_gather_feat 

from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
import numpy as np

INF = 1e8
EPS = 1e-12
PI = np.pi


@HEADS.register_module()
class M3D_HeatMap_head(nn.Module):
    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 bbox3d_code_size=7,
                 num_kpt=9,
                 num_alpha_bins=12,
                 max_objs=30,
                 vector_regression_level=1,
                 pred_bbox2d=True,
                 loss_center_heatmap=None,
                 loss_wh=None,
                 loss_offset=None,
                 loss_center2d_to_3d_offset=None,
                 loss_dim=None,
                 loss_depth_offset=None,
                 loss_alpha_cls=None,
                 loss_alpha_reg=None,
                 use_AN=False,
                 num_AN_affine=10,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(M3D_HeatMap_head, self).__init__()
        assert bbox3d_code_size >= 7
        self.num_classes = num_classes
        self.bbox_code_size = bbox3d_code_size
        self.pred_bbox2d = pred_bbox2d
        self.max_objs = max_objs
        self.num_kpt = num_kpt
        self.num_alpha_bins = num_alpha_bins
        self.vector_regression_level = vector_regression_level

        # self.use_AN = use_AN
        # self.num_AN_affine = num_AN_affine
        # self.norm = AttnBatchNorm2d if use_AN else nn.BatchNorm2d

        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.depth_offset_head = self._build_head(in_channel, feat_channel, 1)
        self.center2d_to_3d_offset_head = self._build_head(in_channel, feat_channel, 2)
        self.dim_head = self._build_head(in_channel, feat_channel, 3)
        
        self._build_dir_head(in_channel, feat_channel)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_center2d_to_3d_offset = build_loss(loss_center2d_to_3d_offset)
        self.loss_dim = build_loss(loss_dim)
        if 'Aware' in loss_dim['type']:
            self.dim_aware_in_loss = True
        else:
            self.dim_aware_in_loss = False
        self.loss_depth_offset = build_loss(loss_depth_offset)
        self.loss_alpha_cls = build_loss(loss_alpha_cls)
        self.loss_alpha_reg = build_loss(loss_alpha_reg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channel, feat_channel, out_channel):        
        layer = spconv.SparseSequential(
            spconv.SparseConv3d(in_channel, feat_channel, 3, padding=1), 
            nn.BatchNorm1d(feat_channel), # Note that AttBN here maybe better ?
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(feat_channel, out_channel, 1, indice_key="subm0"),
        )
        return layer

    def _build_dir_head(self, in_channel, feat_channel):
        self.dir_feat = spconv.SparseSequential(
            spconv.SparseConv3d(in_channel, feat_channel, 3, padding=1), 
            nn.BatchNorm1d(feat_channel), # Note that AttBN here maybe better ?
            nn.ReLU(inplace=True),
        )
        self.dir_cls = spconv.SubMConv3d(feat_channel, self.num_alpha_bins, 1, indice_key="subm0")
        self.dir_reg = spconv.SubMConv3d(feat_channel, self.num_alpha_bins, 1, indice_key="subm0")

    def forward(self, feat):
        center_heatmap_pred = self.heatmap_head(feat).dense().sigmoid()
        center_heatmap_pred = torch.clamp(center_heatmap_pred, min=1e-4, max=1 - 1e-4)

        center2d_to_3d_offset_pred = self.center2d_to_3d_offset_head(feat).dense()
        dim_pred = self.dim_head(feat).dense()
        depth_offset_pred = self.depth_offset_head(feat).dense()

        alpha_feat = self.dir_feat(feat)
        alpha_cls_pred = self.dir_cls(alpha_feat).dense()
        alpha_offset_pred = self.dir_reg(alpha_feat).dense()
        return center_heatmap_pred, center2d_to_3d_offset_pred, depth_offset_pred, \
               dim_pred, alpha_cls_pred, alpha_offset_pred


    @force_fp32(apply_to=('center_heatmap_pred', 'center2d_to_3d_offset_pred',
                          'dim_pred', 'alpha_cls_pred',
                          'alpha_offset_pred', 'depth_offset_pred'))
    def loss(self,
             center_heatmap_pred,
             center2d_to_3d_offset_pred,
             depth_offset_pred,
             dim_pred,
             alpha_cls_pred,
             alpha_offset_pred,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             centers2d,
             depths,
             gt_kpts_2d,
             gt_kpts_valid_mask,
             img_metas,
             attr_labels=None,
             proposal_cfg=None,
             gt_bboxes_ignore=None):
        batch_size = center_heatmap_pred.shape[0]

        target_result = self.get_3D_targets(gt_bboxes, gt_labels,
                                         gt_bboxes_3d,
                                         centers2d,
                                         depths,
                                         gt_kpts_2d,
                                         gt_kpts_valid_mask,
                                         center_heatmap_pred.shape,
                                         img_metas[0]['pad_shape'],
                                         img_metas)

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        center2d_to_3d_offset_target = target_result['center2d_to_3d_offset_target']
        dim_target = target_result['dim_target']
        depth_offset_target = target_result['depth_offset_target']
        alpha_cls_target = target_result['alpha_cls_target']
        alpha_offset_target = target_result['alpha_offset_target']

        indices = target_result['indices']
        mask_target = target_result['mask_target']

        # select desired preds and labels based on mask
        # 3d dim
        dim_pred = self.extract_input_from_tensor(dim_pred, indices, mask_target)
        dim_target = self.extract_target_from_tensor(dim_target, mask_target)
        # depth offset
        depth_offset_pred = self.extract_input_from_tensor(depth_offset_pred, indices, mask_target)
        depth_offset_target = self.extract_target_from_tensor(depth_offset_target, mask_target)
        # alpha cls
        alpha_cls_pred = self.extract_input_from_tensor(alpha_cls_pred, indices, mask_target)
        alpha_cls_target = self.extract_target_from_tensor(alpha_cls_target, mask_target).type(torch.long)
        alpha_cls_onehot_target = alpha_cls_target.new_zeros([len(alpha_cls_target), self.num_alpha_bins]).scatter_(
            dim=1, index=alpha_cls_target.view(-1, 1), value=1)
        # alpha offset
        alpha_offset_pred = self.extract_input_from_tensor(alpha_offset_pred, indices, mask_target)
        alpha_offset_pred = torch.sum(alpha_offset_pred * alpha_cls_onehot_target, 1, keepdim=True)
        alpha_offset_target = self.extract_target_from_tensor(alpha_offset_target, mask_target)
        # center2kpt offset
        center2d_to_3d_offset_pred = self.extract_input_from_tensor(center2d_to_3d_offset_pred,
                                                                indices, mask_target)  # B * (num_kpt * 2)
        center2d_to_3d_offset_target = self.extract_target_from_tensor(center2d_to_3d_offset_target, mask_target)

        # 有可能稀疏卷积生成的mask没有覆盖到GT Heatmap 3D 
        # 对于没有覆盖到的，舍弃该样本先！后续需要改进；
        ignore_sample = (dim_pred != torch.tensor([0.0, 0.0, 0.0], dtype=dim_pred.dtype, device=dim_pred.device))
        ignore_sample = torch.any(ignore_sample, dim=1)
        dim_pred = dim_pred[ignore_sample, ...]
        dim_target = dim_target[ignore_sample, ...]

        depth_offset_pred = depth_offset_pred[ignore_sample, ...]
        depth_offset_target = depth_offset_target[ignore_sample, ...]
        center2d_to_3d_offset_pred = center2d_to_3d_offset_pred[ignore_sample, ...]
        center2d_to_3d_offset_target = center2d_to_3d_offset_target[ignore_sample, ...]
        alpha_cls_pred = alpha_cls_pred[ignore_sample, ...]
        alpha_cls_onehot_target = alpha_cls_onehot_target[ignore_sample, ...]
        alpha_offset_pred = alpha_offset_pred[ignore_sample, ...]
        alpha_offset_target = alpha_offset_target[ignore_sample, ...]

        # calculate loss
        loss_center_heatmap = self.loss_center_heatmap(center_heatmap_pred, center_heatmap_target)
        if self.dim_aware_in_loss:
            loss_dim = self.loss_dim(dim_pred, dim_target, dim_pred)
        else:
            loss_dim = self.loss_dim(dim_pred, dim_target)

        loss_depth_offset = self.loss_depth_offset(depth_offset_pred, depth_offset_target)

        loss_center2d_to_3d_offset = self.loss_center2d_to_3d_offset(center2d_to_3d_offset_pred, center2d_to_3d_offset_target)

        if mask_target.sum() > 0:
            loss_alpha_cls = self.loss_alpha_cls(alpha_cls_pred, alpha_cls_onehot_target)
        else:
            loss_alpha_cls = 0.0
        loss_alpha_reg = self.loss_alpha_reg(alpha_offset_pred, alpha_offset_target)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_dim=loss_dim,
            loss_center2d_to_3d_offset=loss_center2d_to_3d_offset,
            loss_alpha_cls=loss_alpha_cls,
            loss_alpha_reg=loss_alpha_reg,
            loss_depth_offset=loss_depth_offset,
        )


    def get_3D_targets(self, gt_bboxes, gt_labels,
                    gt_bboxes_3d,
                    centers2d,
                    depths,
                    gt_kpts_2d,
                    gt_kpts_valid_mask,
                    feat_shape, img_shape, 
                    img_metas, depth_rel=6):
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w, feat_d = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        calibs = []

        # objects as 2D center points
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w, feat_d])

        # 2D attributes
        wh_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])
        offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])

        # 3D attributes
        dim_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 3])
        alpha_cls_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        alpha_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        depth_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])

        ## 2D-3D kpt heatmap and offset
        center2d_to_3d_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])

        # indices
        indices = gt_bboxes[-1].new_zeros([bs, self.max_objs]).type(torch.cuda.LongTensor)

        # masks
        mask_target = gt_bboxes[-1].new_zeros([bs, self.max_objs])

        for batch_id in range(bs):
            img_meta = img_metas[batch_id]
            cam_p2 = img_meta['cam_intrinsic']

            gt_bbox = gt_bboxes[batch_id]
            calibs.append(cam_p2)
            if len(gt_bbox) < 1:
                continue
            gt_label = gt_labels[batch_id]
            gt_bbox_3d = gt_bboxes_3d[batch_id]
            if not isinstance(gt_bbox_3d, torch.Tensor):
                gt_bbox_3d = gt_bbox_3d.tensor.to(gt_bbox.device)

            depth = depths[batch_id]

            gt_kpt_2d = gt_kpts_2d[batch_id]
            # gt_kpt_valid_mask = gt_kpts_valid_mask[batch_id]

            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            gt_kpt_2d = gt_kpt_2d.reshape(-1, self.num_kpt, 2)
            gt_kpt_2d[:, :, 0] *= width_ratio
            gt_kpt_2d[:, :, 1] *= height_ratio

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio

                dim = gt_bbox_3d[j][3: 6]
                alpha = gt_bbox_3d[j][6]
                gt_kpt_2d_single = gt_kpt_2d[j]  # (9, 2)

                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                
                depth_bin_index = bin_depths(depth[j])

                gen_gaussian_3D_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int, depth_bin_index], radius, depth_rel)

                indices[batch_id, j] = cty_int * feat_w * feat_d + ctx_int * feat_d + depth_bin_index

                wh_target[batch_id, j, 0] = scale_box_w
                wh_target[batch_id, j, 1] = scale_box_h
                offset_target[batch_id, j, 0] = ctx - ctx_int
                offset_target[batch_id, j, 1] = cty - cty_int

                dim_target[batch_id, j] = dim
                depth_offset_target[batch_id, j] = depth[j] - bin_to_depth(depth_bin_index)

                alpha_cls_target[batch_id, j], alpha_offset_target[batch_id, j] = self.angle2class(alpha)

                mask_target[batch_id, j] = 1

                # kpt refers to c2d_3d
                kpt = gt_kpt_2d_single[8]
                kptx_int, kpty_int = kpt.int()
                kptx, kpty = kpt

                center2d_to_3d_offset_target[batch_id, j, 0] = kptx - ctx_int
                center2d_to_3d_offset_target[batch_id, j, 1] = kpty - cty_int

        mask_target = mask_target.type(torch.bool)

        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            center2d_to_3d_offset_target=center2d_to_3d_offset_target,
            dim_target=dim_target,
            depth_offset_target=depth_offset_target,
            alpha_cls_target=alpha_cls_target,
            alpha_offset_target=alpha_offset_target,
            indices=indices,
            mask_target=mask_target
        )

        return target_result

    @staticmethod
    def extract_input_from_tensor(input, ind, mask):
        input = transpose_and_gather_feat(input, ind)
        return input[mask]

    @staticmethod
    def extract_target_from_tensor(target, mask):
        return target[mask]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class and residual. '''
        angle = angle % (2 * PI)
        assert (angle >= 0 and angle <= 2 * PI)
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * PI)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, cls, residual):
        ''' Inverse function to angle2class. '''
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        angle_center = cls * angle_per_class
        angle = angle_center + residual
        return angle

    def decode_alpha_multibin(self, alpha_cls, alpha_offset):
        alpha_score, cls = alpha_cls.max(dim=-1)
        cls = cls.unsqueeze(2)
        alpha_offset = alpha_offset.gather(2, cls)
        alpha = self.class2angle(cls, alpha_offset)

        alpha[alpha > PI] = alpha[alpha > PI] - 2 * PI
        alpha[alpha < -PI] = alpha[alpha < -PI] + 2 * PI
        return alpha
    
    def get_bboxes(self,
                   center_heatmap_pred,
                   center2d_to_3d_offset_pred_pred,
                   depth_offset_pred,
                   dim_pred,
                   alpha_cls_pred,
                   alpha_offset_pred,
                   img_metas,
                   rescale=False):
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        box_type_3d = img_metas[0]['box_type_3d']

        batch_det_bboxes, batch_det_bboxes_3d, batch_labels = self.decode_3D_heatmap(
            center_heatmap_pred,
            center2d_to_3d_offset_pred_pred,
            dim_pred,
            alpha_cls_pred,
            alpha_offset_pred,
            depth_offset_pred,
            img_metas[0]['pad_shape'][:2],
            img_metas[0]['cam_intrinsic'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel,
            thresh=self.test_cfg.thresh)

        if rescale:
            batch_det_bboxes[..., :4] /= batch_det_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        det_results = [
            [box_type_3d(batch_det_bboxes_3d,
                         box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5)),
             batch_det_bboxes[:, -1],
             batch_labels,
             batch_det_bboxes,
             ]
        ]
        return det_results

    def decode_3D_heatmap(self,
                       center_heatmap_pred,
                       center2d_to_3d_offset_pred_pred,
                       dim_pred,
                       alpha_cls_pred,
                       alpha_offset_pred,
                       depth_offset_pred,
                       img_shape,
                       camera_intrinsic,
                       k=100,
                       kernel=3,
                       thresh=0.4):
        batch, cat, height, width, length = center_heatmap_pred.shape
        assert batch == 1
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, ys, xs, zs = get_topk_from_3D_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        topk_xs = xs + offset[..., 0]
        topk_ys = ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)  # (b, k, 5)

        # decode 3D prediction
        dim = transpose_and_gather_feat(dim_pred, batch_index)
        alpha_cls = transpose_and_gather_feat(alpha_cls_pred, batch_index)
        alpha_offset = transpose_and_gather_feat(alpha_offset_pred, batch_index)
        depth_offset_pred = transpose_and_gather_feat(depth_offset_pred, batch_index)
        depth = bin_to_depth(zs) + depth_offset_pred

        center2d_to_3d_offset_pred = transpose_and_gather_feat(center2d_to_3d_offset_pred_pred, batch_index)
        center2d_to_3d_offset_pred = center2d_to_3d_offset_pred.view(batch, k, 2)
        center2d_to_3d_offset_pred[..., ::2] += xs.view(batch, k, 1).expand(batch, k, 1)
        center2d_to_3d_offset_pred[..., 1::2] += ys.view(batch, k, 1).expand(batch, k, 1)

        kpts = center2d_to_3d_offset_pred

        kpts[..., 0] *= (inp_w / width)
        kpts[..., 1] *= (inp_h / height)

        # 1. decode alpha
        alpha = self.decode_alpha_multibin(alpha_cls, alpha_offset)  # (b, k, 1)

        # 1.5 get projected center
        center2d = kpts  # (b, k, 2)

        # 2. recover rotY
        rot_y = self.recover_rotation(kpts, alpha, camera_intrinsic)  # (b, k, 3)

        # 2.5 recover box3d_center from center2d and depth
        center3d = torch.cat([center2d, depth], dim=-1).squeeze(0)
        center3d = self.pts2Dto3D(center3d, np.array(camera_intrinsic)).unsqueeze(0)

        # 3. compose 3D box
        batch_bboxes_3d = torch.cat([center3d, dim, rot_y], dim=-1)

        mask = batch_bboxes[..., -1] > thresh
        batch_bboxes = batch_bboxes[mask]
        batch_bboxes_3d = batch_bboxes_3d[mask]
        batch_topk_labels = batch_topk_labels[mask]

        return batch_bboxes, batch_bboxes_3d, batch_topk_labels

    def recover_rotation(self, kpts, alpha, calib):
        device = kpts.device
        calib = torch.tensor(calib).type(torch.FloatTensor).to(device).unsqueeze(0)

        si = torch.zeros_like(kpts[:, :, 0:1]) + calib[:, 0:1, 0:1]
        rot_y = alpha + torch.atan2(kpts[:, :, 0:1] - calib[:, 0:1, 2:3], si)

        while (rot_y > PI).any():
            rot_y[rot_y > PI] = rot_y[rot_y > PI] - 2 * PI
        while (rot_y < -PI).any():
            rot_y[rot_y < -PI] = rot_y[rot_y < -PI] + 2 * PI

        return rot_y

    @staticmethod
    def pts2Dto3D(points, view):
        """
        Args:
            points (torch.Tensor): points in 2D images, [N, 3], \
                3 corresponds with x, y in the image and depth.
            view (np.ndarray): camera instrinsic, [3, 3]

        Returns:
            torch.Tensor: points in 3D space. [N, 3], \
                3 corresponds with x, y, z in 3D space.
        """
        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[1] == 3

        points2D = points[:, :2]
        depths = points[:, 2].view(-1, 1)
        unnorm_points2D = torch.cat([points2D * depths, depths], dim=1)

        viewpad = torch.eye(4, dtype=points2D.dtype, device=points2D.device)
        viewpad[:view.shape[0], :view.shape[1]] = points2D.new_tensor(view)
        inv_viewpad = torch.inverse(viewpad).transpose(0, 1)

        # Do operation in homogenous coordinates.
        nbr_points = unnorm_points2D.shape[0]
        homo_points2D = torch.cat(
            [unnorm_points2D,
             points2D.new_ones((nbr_points, 1))], dim=1)
        points3D = torch.mm(homo_points2D, inv_viewpad)[:, :3]

        return points3D

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      centers2d=None,
                      depths=None,
                      gt_kpts_2d=None,
                      gt_kpts_valid_mask=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        assert gt_labels is not None
        assert attr_labels is None
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_bboxes_3d,
                              gt_labels_3d, centers2d, depths, gt_kpts_2d, gt_kpts_valid_mask,
                              img_metas, attr_labels)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if proposal_cfg is None:
            return losses
        else:
            raise NotImplementedError