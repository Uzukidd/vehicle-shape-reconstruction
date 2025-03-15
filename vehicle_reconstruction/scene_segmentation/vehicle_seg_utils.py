# from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from vehicle_reconstruction import *
from cudaext.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

from vehicle_reconstruction.scene_segmentation.loss_utils import shape_regular

class vehicle_object(object):

    def __init__(self, pts: torch.Tensor, bbox: torch.Tensor, k=5) -> None:
        """
            pts: [N, 3]
            bbox: [7]
        """
        self.pts = pts.detach().clone()
        self.bbox = bbox.detach().clone()
        # self.vehicle: "vehicle" = None
        # self.vehicle_latent = torch.zeros(
        #     (1, k), requires_grad=True, device=pts.device)
        # [1, K]

        # self.translation = torch.tensor([
        #     bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[6].item()
        # ], requires_grad=True, device=pts.device)

    # def centroid(self):
    #     pts_centroid = self.pts[:, :3] - self.translation[None, :3]
    #     pts_centroid = torch.matmul(pts_centroid, pts_centroid.new_tensor(
    #         [[torch.cos(-self.translation[3]),
    #           -torch.sin(-self.translation[3]), 0.0],
    #          [torch.sin(-self.translation[3]),
    #           torch.cos(-self.translation[3]), 0.0],
    #          [0.0, 0.0, 1.0],
    #          ]
    #     ).T)

    #     # undifferentiable
    #     bbox_centroid = torch.zeros_like(self.bbox)
    #     bbox_centroid[3:6] = self.bbox[3:6]

    #     return pts_centroid, bbox_centroid

    # def reconstruct_at_scene(self, vehi_reconstructor, standard_bbox, show_rooftop: bool = False):
    #     self.reconstruct_vehicle(vehi_reconstructor)
    #     vehi = self.vehicle.to_mesh()
    #     vehicle_mesh = vehi.to_trimesh()

    #     vehicle_mesh.vertices *= 0.1
    #     vehicle_mesh.vertices = vehicle_mesh.vertices[:, [2, 0, 1]]

    #     vehicle_mesh.vertices -= np.array([standard_bbox[2]/2.0,
    #                                        standard_bbox[0]/2.0,
    #                                        self.bbox[5].cpu() / 2.0])
    #     vehicle_mesh.vertices = np.matmul(vehicle_mesh.vertices, np.array(
    #         [[np.cos(self.bbox[6].cpu()),
    #           -np.sin(self.bbox[6].cpu()), 0.0],
    #          [np.sin(self.bbox[6].cpu()),
    #           np.cos(self.bbox[6].cpu()), 0.0],
    #          [0.0, 0.0, 1.0],
    #          ]
    #     ).T)

    #     vehicle_mesh.vertices += np.array([self.bbox[0].cpu(),
    #                                        self.bbox[1].cpu(),
    #                                        self.bbox[2].cpu()])

    #     if show_rooftop:
    #         max_y = vehicle_mesh.vertices[:, 2].max()
    #         rooftop_idx = ((max_y - vehicle_mesh.vertices[:, 2]) < 0.2)
    #         # rooftop_vertices = self.vertices[rooftop_idx]
    #         # rooftop_vertices, rooftop_idx = vehi.rooftop_approximate(
    #         #     return_idx=True)
    #         return vehicle_mesh, rooftop_idx
    #     else:
    #         return vehicle_mesh

    # def reconstruct_vehicle(self, reconstructor: vehicle_reconstructor):
    #     self.vehicle = reconstructor.decode(self.vehicle_latent)[0]

    # def prepare_training_data(self, reconstructor, padding: bool = True):
    #     voxel: torch.Tensor = reconstructor.decode_aux(self.vehicle_latent)[0]
    #     voxel = voxel.permute([2, 0, 1])  # [w, h, l] -> [l, w, h]

    #     #  padding
    #     if padding:
    #         voxel = F.pad(voxel, (1, 1, 1, 1, 1, 1), "constant", 0.2)

    #     pts_centroid, _ = self.centroid()  # [N, 3] world coordinate

    #     return voxel, pts_centroid, self.bbox[5].item()


class point_cloud_scene(object):

    def __init__(self, pts: torch.Tensor, gt_boxes: torch.Tensor, vehi_reconstructor: vehicle_reconstructor) -> None:
        """
            pts: [N, 4]
            gt_boxes: [N, 7]
        """
        self.pts: torch.Tensor = pts.detach().clone()
        self.gt_boxes: torch.Tensor = gt_boxes.detach().clone()
        self.vehicles: list[vehicle_object] = None
        self.vehi_reconstructor: vehicle_reconstructor = vehi_reconstructor
        # self.estimate_loss_fun: pose_estimate_loss = pose_estimate_loss(
        #     vehi_reconstructor.sampling_space[2],
        #     vehi_reconstructor.sampling_space[0],
        #     vehi_reconstructor.sampling_space[1],
        #     vehi_reconstructor.global_res)
        # self.shape_regular_fun: shape_regular = shape_regular(
        #     vehi_reconstructor.S)

    def vehicle_seg(self):
        self.vehicles = []

        self.pts_assign = points_in_boxes_gpu(
            self.pts[:, :3].view(1, -1, 3), self.gt_boxes.view(1, -1, 7)).squeeze(dim=0)
        for bbox_idx in range(self.gt_boxes.size(0)):

            bbox_mask = (self.pts_assign == bbox_idx)
            
            if bbox_mask.numel() < 10 or bbox_mask.sum() < 10:
                self.vehicles.append(None)
                continue

            vehicle = vehicle_object(pts=self.pts[bbox_mask],
                                     bbox=self.gt_boxes[bbox_idx])
            self.vehicles.append(vehicle)

    def get_vehicles(self):
        if self.vehicles is None:
            self.vehicle_seg()

        return self.vehicles
    
    def get_vehicles_traning_batch(self):
        if self.vehicles is None:
            self.vehicle_seg()

        for vehi in self.vehicles:
            if vehi is None:
                continue
            
            for _ in range(iter):
                voxel, pts_centroid, height = vehi.prepare_training_data(
                    self.vehi_reconstructor)
                shape_loss = self.estimate_loss_fun(
                    voxel, pts_centroid, height)
                shape_regulation = self.shape_regular_fun(
                    vehi.vehicle_latent)

                total_loss = (16.6 / self.vehi_reconstructor.global_res) * \
                    shape_loss + shape_regulation
                total_loss.backward()
                
                vehi.vehicle_latent = vehi.vehicle_latent - vehi.vehicle_latent.grad
                vehi.vehicle_latent = vehi.vehicle_latent.clone().detach()
                vehi.vehicle_latent.requires_grad_(True)

    def pose_estimate(self, iter=5):
        if self.vehicles is None:
            self.vehicle_seg()

        for vehi in self.vehicles:
            if vehi is None:
                continue
            
            for _ in range(iter):
                voxel, pts_centroid, height = vehi.prepare_training_data(
                    self.vehi_reconstructor)
                shape_loss = self.estimate_loss_fun(
                    voxel, pts_centroid, height)
                shape_regulation = self.shape_regular_fun(
                    vehi.vehicle_latent)

                total_loss = (16.6 / self.vehi_reconstructor.global_res) * \
                    shape_loss + shape_regulation
                total_loss.backward()
                
                vehi.vehicle_latent = vehi.vehicle_latent - vehi.vehicle_latent.grad
                vehi.vehicle_latent = vehi.vehicle_latent.clone().detach()
                vehi.vehicle_latent.requires_grad_(True)
                
                # vehi.translation = vehi.translation - vehi.translation.grad
                # vehi.translation = vehi.translation.clone().detach()
                # vehi.translation.requires_grad_(True)

            # if vehi.translation.grad is not None:
            #     vehi.translation = vehi.translation - vehi.translation.grad
            #     vehi.translation = vehi.translation.clone().detach()
            #     vehi.translation.requires_grad_(True)
            # print(vehi.vehicle_latent.grad)
            # print(vehi.translation.grad)
