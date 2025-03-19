# from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vehicle_reconstruction import *
from cudaext.ops.trilinear_interpolate.trilinear_interpolate_utils import Trilinear_interpolation_cuda

class shape_regular(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, latent: torch.Tensor, eigens:torch.Tensor):
        """
            latent: [B, K]
            eigens: [K]
        """
        return 0.5 * (latent/eigens).square().mean()
    
class vehicles_roi_pooling(nn.Module):
    
    def __init__(self,
                 sample_count:int = 100,
                 method:str = "pad_zeros") -> None:
        super().__init__()
        self.sample_count = sample_count
        self.method = method
        
    def forward(self, vehicle_feats: list[torch.Tensor]):
        """
            vehicle_feats: list([N, F])
        """
        batch_size = vehicle_feats.__len__()
        pts_reduced_size = self.sample_count
        feats_size = vehicle_feats[0].size(1)
        feats_reduced_tensor = []
        if self.method == "pad_zeros":
            for feats_tensor in vehicle_feats:
                feats_size_offset = self.sample_count - feats_tensor.size(0)
                if feats_size_offset > 0:
                    feats_reduced_tensor.append(F.pad(feats_tensor, (0, 0, 0, feats_size_offset), mode="constant", value=0))
                else:
                    feats_reduced_tensor.append(feats_tensor[:self.sample_count])
                    
        feats_reduced_tensor = torch.stack(feats_reduced_tensor)
        return feats_reduced_tensor
    
class pose_estimate_loss_batch(nn.Module):
    """
        loss from https://www.vision.rwth-aachen.de/media/papers/EngelmannGCPR16_SZa4QgP.pdf
    """

    def __init__(self) -> None:
        super().__init__()
        self.trilinear_interpolate:Trilinear_interpolation_cuda = Trilinear_interpolation_cuda()

    def forward(self, tsdf_grid: torch.Tensor, pts_centroid: torch.Tensor, grid_unit: torch.Tensor):
        """
            tsdf_grid: [B, L, W, H]
            pooled pts_centroid: [B, N, 3] -> (0, L * UNIT) x (0, W * UNIT) x (0, H * UNIT)
            height: [B, N]
        """
        batch_size, grid_l, grid_w, grid_h = tsdf_grid.size()

        # compute grid position
        pts_centroid_flatten = pts_centroid.view(-1, 3) # [B * N, 3]
        with torch.no_grad():
            x_min = torch.floor(
                pts_centroid_flatten[:, 0] / grid_unit[0])
            x_max = x_min + 1

            y_min = torch.floor(
                pts_centroid_flatten[:, 1] / grid_unit[1])
            y_max = y_min + 1

            z_min = torch.floor(
                pts_centroid_flatten[:, 2] / grid_unit[2])
            z_max = z_min + 1

        # truncated to [0.0, grid_unit]
        pts_centroid_flatten[:, 0] -= (x_min * grid_unit[0]) # [B * N]
        pts_centroid_flatten[:, 1] -= (y_min * grid_unit[1])
        pts_centroid_flatten[:, 2] -= (z_min * grid_unit[2])

        # project to [-1.0, 1.0]
        pts_centroid_flatten[:, :] *= (2 / grid_unit) # [B * N, 3]
        pts_centroid_flatten[:, :] -= 1.0

        # limit x, y, z into grid boundary
        x_min = x_min.clamp(0, grid_l-1).long() # [B * N]
        y_min = y_min.clamp(0, grid_w-1).long()
        z_min = z_min.clamp(0, grid_h-1).long()
        x_max = x_max.clamp(0, grid_l-1).long()
        y_max = y_max.clamp(0, grid_w-1).long()
        z_max = z_max.clamp(0, grid_h-1).long()
        
        # batch_mask = torch.arange(batch_size).unsqueeze(1).expand_as(x_min)
        batch_mask = torch.arange(batch_size).unsqueeze(1).expand(-1, pts_centroid.size(1)).flatten() # [B * N]

        feature_stack = torch.stack([
            tsdf_grid[batch_mask, x_max, y_max, z_max],  # [B * N]
            tsdf_grid[batch_mask, x_max, y_max, z_min],
            tsdf_grid[batch_mask, x_max, y_min, z_max],
            tsdf_grid[batch_mask, x_max, y_min, z_min],

            tsdf_grid[batch_mask, x_min, y_max, z_max],
            tsdf_grid[batch_mask, x_min, y_max, z_min],
            tsdf_grid[batch_mask, x_min, y_min, z_max],
            tsdf_grid[batch_mask, x_min, y_min, z_min],
        ], dim=1).unsqueeze(2).contiguous() # [B * N, 8]

        sdf_val = self.trilinear_interpolate.apply(feature_stack, pts_centroid_flatten)
        # sdf_val = trilinear_interpolation_cpu(feature_stack, pts_centroid)
        loss = F.huber_loss(sdf_val, torch.zeros_like(sdf_val))

        return loss