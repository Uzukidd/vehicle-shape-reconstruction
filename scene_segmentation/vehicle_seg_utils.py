import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..vehicle_reconstruction import *
from cudaext.ops.trilinear_interpolate.trilinear_interpolate_utils import Trilinear_interpolation_cuda, trilinear_interpolation_cpu
from cudaext.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu


class shape_regular(nn.Module):
    def __init__(self, eigens: torch.Tensor) -> None:
        super().__init__()
        self.eigens = eigens.detach().clone().view(1, -1)  # [K]

    def forward(self, latent: torch.Tensor):
        """
            latent: [1, K]
        """
        assert latent.size(0) == 1, "batch computation is not yet supported."
        return 0.5 * (latent/self.eigens).square().sum()


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
        assert pts_centroid.size().__len__() < 3, "batch computation is not yet supported."

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


class vehicle_object(object):

    def __init__(self, pts: torch.Tensor, bbox: torch.Tensor, k=5) -> None:
        """
            pts: [N, 3]
            bbox: [7]
        """
        self.pts = pts
        self.bbox = bbox
        self.vehicle: "vehicle" = None
        self.vehicle_latent = torch.zeros(
            (1, k), requires_grad=True, device=pts.device)
        # [1, K]

        self.translation = torch.tensor([
            bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[6].item()
        ], requires_grad=True, device=pts.device)

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

    def reconstruct_at_scene(self, vehi_reconstructor, standard_bbox, show_rooftop: bool = False):
        self.reconstruct_vehicle(vehi_reconstructor)
        vehi = self.vehicle.to_mesh()
        vehicle_mesh = vehi.to_trimesh()
        vehicle_mesh.vertices *= 0.1
        vehicle_mesh.vertices = vehicle_mesh.vertices[:, [2, 0, 1]]

        vehicle_mesh.vertices -= np.array([standard_bbox[2]/2.0,
                                           standard_bbox[0]/2.0,
                                           self.bbox[5].cpu() / 2.0])
        vehicle_mesh.vertices = np.matmul(vehicle_mesh.vertices, np.array(
            [[np.cos(self.bbox[6].cpu()),
              -np.sin(self.bbox[6].cpu()), 0.0],
             [np.sin(self.bbox[6].cpu()),
              np.cos(self.bbox[6].cpu()), 0.0],
             [0.0, 0.0, 1.0],
             ]
        ).T)

        vehicle_mesh.vertices += np.array([self.bbox[0].cpu(),
                                           self.bbox[1].cpu(),
                                           self.bbox[2].cpu()])
        if show_rooftop:
            rooftop_vertices, rooftop_idx = vehi.to_mesh().rooftop_approximate(
                return_idx=True)
            return vehicle_mesh, rooftop_idx
        else:
            return vehicle_mesh

    def reconstruct_vehicle(self, reconstructor: "vehicle_reconstructor"):
        self.vehicle = reconstructor.decode(self.vehicle_latent)[0]

    def prepare_training_data(self, reconstructor, padding: bool = True):
        voxel: torch.Tensor = reconstructor.decode_aux(self.vehicle_latent)[0]
        voxel = voxel.permute([2, 0, 1])  # [w, h, l] -> [l, w, h]

        #  padding
        if padding:
            voxel = F.pad(voxel, (1, 1, 1, 1, 1, 1), "constant", 0.2)

        pts_centroid, _ = self.centroid()  # [N, 3] world coordinate

        return voxel, pts_centroid, self.bbox[5].item()


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
            vehi_reconstructor.sampling_space[1],
            vehi_reconstructor.global_res)
        self.shape_regular_fun: shape_regular = shape_regular(
            vehi_reconstructor.S)

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

    def pose_estimate(self, iter=5):
        if self.vehicles is None:
            self.vehicle_seg()

        for vehi in self.vehicles:
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

            # if vehi.translation.grad is not None:
            #     vehi.translation = vehi.translation - vehi.translation.grad
            #     vehi.translation = vehi.translation.clone().detach()
            #     vehi.translation.requires_grad_(True)
            # print(vehi.vehicle_latent.grad)
            # print(vehi.translation.grad)
