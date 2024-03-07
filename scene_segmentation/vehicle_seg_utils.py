import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cudaext.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu


class pose_estimate_loss(nn.Module):

    def __init__(self, length: int, width: int, height: int) -> None:
        super().__init__()
        self.length = length
        self.width = width
        self.height = height

    def forward(self, voxels: torch.Tensor, pts_centroid: torch.Tensor, height_gt: torch.Tensor):
        """
            voxels: [B, L, W, H]
            pts_centroid: [B, N, 3]
            height: [B, N, 1]
        """
        # assert pts_centroid.size().__len__() < 3, "batch computation is not yet supported."
        pts_centroid[:, :, 2] += height_gt
        pts_centroid[:, :, 2] -= self.height

        pts_centroid[:, :, 0] /= (self.length/2.0)
        pts_centroid[:, :, 1] /= (self.width/2.0)
        pts_centroid[:, :, 2] /= (self.height/2.0)

        F.grid_sample(pts_centroid, voxels,
                      padding_mode="border", align_corners=True)


class vehicle_object(object):

    def __init__(self, pts: torch.Tensor, bbox: torch.Tensor, k=5) -> None:
        """
            pts: [N, 3]
            bbox: [7]
        """
        self.pts = pts
        self.bbox = bbox
        self.vehicle: "vehicle" = None
        self.vehicle_latent = torch.randn((1, k), requires_grad=True).cuda()

        self.translation = torch.tensor([
            bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[6].item()
        ], requires_grad=True).cuda()

    def centroid(self):
        pts_centroid = self.pts[:, :3] - self.translation[None, :3]
        pts_centroid = torch.matmul(pts_centroid, pts_centroid.new_tensor(
            [[torch.cos(-self.translation[3]),
              -torch.sin(-self.translation[3]), 0.0],
             [torch.sin(-self.translation[3]),
              torch.cos(-self.translation[3]), 0.0],
             [0.0, 0.0, 1.0],
             ]
        ).T)

        # undifferentiable
        bbox_centroid = torch.zeros_like(self.bbox)
        bbox_centroid[3:6] = self.bbox[3:6]

        return pts_centroid, bbox_centroid

    def reconstruct_vehicle(self, reconstructor: "vehicle_reconstructor"):
        self.vehicle = reconstructor.decode(self.vehicle_latent)[0]

    def prepare_training_data(self, reconstructor):
        voxel: torch.Tensor = reconstructor.decode_aux(self.vehicle_latent)
        voxel = voxel.permute([2, 0, 1])  # [l, w, h]

        pts_centroid, _ = self.centroid()  # [N, 3] world coordinate

        return voxel, pts_centroid


class point_cloud_scene(object):

    def __init__(self, pts: torch.Tensor, gt_boxes: torch.Tensor, vehi_reconstructor: "vehicle_reconstructor") -> None:
        """
            pts: [N, 4]
            gt_boxes: [N, 7]
        """
        self.pts: torch.Tensor = pts
        self.gt_boxes: torch.Tensor = gt_boxes
        self.vehicles: list[vehicle_object] = None
        self.vehi_reconstructor: "vehicle_reconstructor" = vehi_reconstructor
        self.estimate_loss_fun: pose_estimate_loss = pose_estimate_loss(
            vehi_reconstructor.sampling_space[2],
            vehi_reconstructor.sampling_space[0],
            vehi_reconstructor.sampling_space[1])

    def vehicle_seg(self):
        self.vehicles = []

        self.pts_assign = points_in_boxes_gpu(
            self.pts[:, :3].view(1, -1, 3), self.gt_boxes.view(1, -1, 7)).squeeze(dim=0)
        for bbox_idx in range(self.gt_boxes.size(0)):

            bbox_mask = (self.pts_assign == bbox_idx)

            if not bbox_mask.any():
                continue

            vehicle = vehicle_object(pts=self.pts[bbox_mask],
                                     bbox=self.gt_boxes[bbox_idx])
            self.vehicles.append(vehicle)

    def get_vehicles(self):
        if self.vehicles is None:
            self.vehicle_seg()

        return self.vehicles
