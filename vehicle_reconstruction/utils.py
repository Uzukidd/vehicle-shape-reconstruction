import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model_bbox(vertices: np.ndarray):
    xmin, xmax = vertices[:, 0].min(), vertices[:, 0].max()
    ymin, ymax = vertices[:, 1].min(), vertices[:, 1].max()
    zmin, zmax = vertices[:, 2].min(), vertices[:, 2].max()
    length = xmax - xmin
    width = ymax - ymin
    height = zmax - zmin

    return (length, width, height, xmin, xmax, ymin, ymax, zmin, zmax)


class pose_estimate_loss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, voxels: torch.Tensor, translation: torch.Tensor, pts: torch.Tensor, grid_lwh: torch.Tensor):
        """
            voxels: [B, L, W, H]
            translation: [B, 3]
            pts: [B, N, 3]
            grid_lwh: [B, N, 3]
        """
        # assert pts.size().__len__() < 3, "batch computation is not yet supported."
        pts_translated = pts - translation[:, None, :]
        
        pts_translated
