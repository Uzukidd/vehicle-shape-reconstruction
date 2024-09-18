# from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vehicle_reconstruction import *
from cudaext.ops.trilinear_interpolate.trilinear_interpolate_utils import Trilinear_interpolation_cuda

class shape_regular(nn.Module):
    def __init__(self, eigens: torch.Tensor) -> None:
        super().__init__()
        self.eigens = eigens.detach().clone().view(1, -1)  # [K]

    def forward(self, latent: torch.Tensor):
        """
            latent: [1, K]
        """
        assert latent.size(0) == 1, "batch forwarding is not yet supported."
        return 0.5 * (latent/self.eigens).square().sum()
    
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

    def __init__(self, length: float,
                 width: float,
                 height: float,
                 grid_res: float = 0.1) -> None:

        super().__init__()
        self.length:float = length
        self.width:float = width
        self.height:float = height
        self.grid_res:float = grid_res
        
        self.trilinear_interpolate:Trilinear_interpolation_cuda = Trilinear_interpolation_cuda()

    def forward(self, voxels: torch.Tensor, pts_centroid: torch.Tensor, height_gt: torch.Tensor):
        """
            voxels: [B, L, W, H]
            pts_centroid: [B, N, 3]
            height: [B, N]
        """
        batch_size, grid_l, grid_w, grid_h = voxels.size(0), voxels.size(1), voxels.size(2), voxels.size(3)

        # center position
        pts_centroid[:, :, 0] += (self.length/2.0)  # [B, N]
        pts_centroid[:, :, 1] += (self.width/2.0)   # [B, N]
        pts_centroid[:, :, 2] += (height_gt/2.0)    # [B, N]

        # compute grid position
        pts_centroid_squeezed = pts_centroid.view(-1, 3) # [B, N]
        with torch.no_grad():
            x_min = torch.floor(
                pts_centroid_squeezed[:, 0] / self.grid_res)
            x_max = x_min + 1

            y_min = torch.floor(
                pts_centroid_squeezed[:, 1] / self.grid_res)
            y_max = y_min + 1

            z_min = torch.floor(
                pts_centroid_squeezed[:, 2] / self.grid_res)
            z_max = z_min + 1

        pts_centroid[:, :, 0] -= (x_min * self.grid_res) # [B, N]
        pts_centroid[:, :, 1] -= (y_min * self.grid_res)
        pts_centroid[:, :, 2] -= (z_min * self.grid_res)

        # project to [-1.0, 1.0]
        pts_centroid[:, :, :] *= (2 / self.grid_res) # [B, N, 3]
        pts_centroid[:, :, :] -= 1.0

        # limit x, y, z into grid boundary
        x_min = x_min.clamp_(0, grid_l-1).long() # [B, N]
        y_min = y_min.clamp_(0, grid_w-1).long()
        z_min = z_min.clamp_(0, grid_h-1).long()
        x_max = x_max.clamp_(0, grid_l-1).long()
        y_max = y_max.clamp_(0, grid_w-1).long()
        z_max = z_max.clamp_(0, grid_h-1).long()
        
        batch_mask = torch.arange(batch_size).unsqueeze(1).expand_as(x_min)

        feature_stack = torch.stack([
            voxels[batch_mask, x_max, y_max, z_max],  # [B, N]
            voxels[batch_mask, x_max, y_max, z_min],
            voxels[batch_mask, x_max, y_min, z_max],
            voxels[batch_mask, x_max, y_min, z_min],

            voxels[batch_mask, x_min, y_max, z_max],
            voxels[batch_mask, x_min, y_max, z_min],
            voxels[batch_mask, x_min, y_min, z_max],
            voxels[batch_mask, x_min, y_min, z_min],
        ], dim=2).contiguous() # [B, N, 8]
        sdf_val = self.trilinear_interpolate.apply(feature_stack, pts_centroid)
        # sdf_val = trilinear_interpolation_cpu(feature_stack, pts_centroid)
        loss = F.huber_loss(sdf_val, torch.zeros_like(sdf_val))

        return loss

class pose_estimate_loss(nn.Module):
    """
        loss from https://www.vision.rwth-aachen.de/media/papers/EngelmannGCPR16_SZa4QgP.pdf
    """

    def __init__(self, length: int,
                 width: int,
                 height: int,
                 grid_res: float = 0.1) -> None:

        super().__init__()
        self.length = length
        self.width = width
        self.height = height
        self.grid_res = grid_res
        self.trilinear_interpolate = Trilinear_interpolation_cuda()

    def forward(self, voxels: torch.Tensor, pts_centroid: torch.Tensor, height_gt: int):
        """
            voxels: [B, L, W, H]
            pts_centroid: [B, N, 3]
            height: [B, N, 1]
            ////////////
            voxels: [L, W, H]
            pts_centroid: [N, 3]
            height: int
        """
        assert pts_centroid.size().__len__() < 3, "batch forwarding is not yet supported."

        grid_l, grid_w, grid_h = voxels.size(0), voxels.size(1), voxels.size(2)

        # center position
        pts_centroid[:, 0] += (self.length/2.0)
        pts_centroid[:, 1] += (self.width/2.0)
        pts_centroid[:, 2] += (height_gt/2.0)

        # compute grid position
        with torch.no_grad():
            x_min = torch.floor(
                pts_centroid[:, 0] / self.grid_res)
            x_max = x_min + 1

            y_min = torch.floor(
                pts_centroid[:, 1] / self.grid_res)
            y_max = y_min + 1

            z_min = torch.floor(
                pts_centroid[:, 2] / self.grid_res)
            z_max = z_min + 1

        pts_centroid[:, 0] -= (x_min * self.grid_res)
        pts_centroid[:, 1] -= (y_min * self.grid_res)
        pts_centroid[:, 2] -= (z_min * self.grid_res)

        # project to [-1.0, 1.0]
        pts_centroid[:, :] *= (2 / self.grid_res)
        pts_centroid[:, :] -= 1.0

        # limit x, y, z into grid boundary
        x_min = x_min.clamp_(0, grid_l-1).long()
        y_min = y_min.clamp_(0, grid_w-1).long()
        z_min = z_min.clamp_(0, grid_h-1).long()
        x_max = x_max.clamp_(0, grid_l-1).long()
        y_max = y_max.clamp_(0, grid_w-1).long()
        z_max = z_max.clamp_(0, grid_h-1).long()

        feature_stack = torch.stack([
            voxels[x_max, y_max, z_max],
            voxels[x_max, y_max, z_min],
            voxels[x_max, y_min, z_max],
            voxels[x_max, y_min, z_min],

            voxels[x_min, y_max, z_max],
            voxels[x_min, y_max, z_min],
            voxels[x_min, y_min, z_max],
            voxels[x_min, y_min, z_min],
        ]).permute([1, 0]).unsqueeze(-1).contiguous()
        sdf_val = self.trilinear_interpolate.apply(feature_stack, pts_centroid)
        # sdf_val = trilinear_interpolation_cpu(feature_stack, pts_centroid)
        loss = F.huber_loss(sdf_val, torch.zeros_like(sdf_val))

        return loss

if __name__ == "__main__":
    roipool = vehicles_roi_pooling()