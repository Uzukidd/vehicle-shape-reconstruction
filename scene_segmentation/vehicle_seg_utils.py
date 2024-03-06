import numpy as np
import torch

from cudaext.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu


class vehicle_object(object):

    def __init__(self, pts: torch.Tensor, bbox: torch.Tensor) -> None:
        """
            pts: [N, 3]
            bbox: [7]
        """
        self.pts = pts
        self.bbox = bbox
        self.vehicle: "vehicle" = None
        self.vehicle_latent = torch.randn((1, 5)).cuda()

    def centroid(self):
        pts_centroid = self.pts[:, :3] - self.bbox[None, :3]
        pts_centroid = torch.matmul(pts_centroid, pts_centroid.new_tensor(
            [[torch.cos(-self.bbox[6]), -torch.sin(-self.bbox[6]), 0.0],
             [torch.sin(-self.bbox[6]), torch.cos(-self.bbox[6]), 0.0],
             [0.0, 0.0, 1.0],
             ]
        ).T)

        bbox_centroid = torch.zeros_like(self.bbox)
        bbox_centroid[3:6] = self.bbox[3:6]

        return pts_centroid, bbox_centroid

    def reconstruct_vehicle(self, reconstructor: "vehicle_reconstructor"):
        self.vehicle = reconstructor.decode(self.vehicle_latent)[0]


class point_cloud_scene(object):

    def __init__(self, pts: torch.Tensor, gt_boxes: torch.Tensor) -> None:
        """
            pts: [N, 4]
            gt_boxes: [N, 7]
        """
        self.pts = pts
        self.gt_boxes = gt_boxes
        self.vehicles = None

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
