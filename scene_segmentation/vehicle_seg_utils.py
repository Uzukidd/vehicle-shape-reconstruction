import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cudaext.ops.trilinear_interpolate.trilinear_interpolate_utils import Trilinear_interpolation_cuda, trilinear_interpolation_cpu
from cudaext.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu


class pose_estimate_loss(nn.Module):

    def __init__(self, length: int, width: int, height: int) -> None:
        super().__init__()
        self.length = length
        self.width = width
        self.height = height
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

        pts_centroid[:, 0] += (self.length/2.0)
        pts_centroid[:, 1] += (self.width/2.0)
        pts_centroid[:, 2] += (height_gt/2.0)

        with torch.no_grad():
            x_min, x_max = torch.floor(
                pts_centroid[:, 0] * 10), torch.floor(pts_centroid[:, 0] * 10) + 1
            y_min, y_max = torch.floor(
                pts_centroid[:, 1] * 10), torch.floor(pts_centroid[:, 1] * 10) + 1
            z_min, z_max = torch.floor(
                pts_centroid[:, 2] * 10), torch.floor(pts_centroid[:, 2] * 10) + 1

        pts_centroid[:, 0] -= x_min/10.0
        pts_centroid[:, 1] -= y_min/10.0
        pts_centroid[:, 2] -= z_min/10.0

        pts_centroid[:, :] *= 20.0
        pts_centroid[:, :] -= 1.0

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

    def reconstruct_at_scene(self, vehi_reconstructor, standard_bbox):
        self.reconstruct_vehicle(vehi_reconstructor)
        vehicle_mesh = self.vehicle.to_trimesh()
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

        return vehicle_mesh

    def reconstruct_vehicle(self, reconstructor: "vehicle_reconstructor"):
        self.vehicle = reconstructor.decode(self.vehicle_latent)[0]

    def prepare_training_data(self, reconstructor):
        voxel: torch.Tensor = reconstructor.decode_aux(self.vehicle_latent)[0]
        voxel = voxel.permute([2, 0, 1])  # [l, w, h]

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

    def pose_estimate(self):
        if self.vehicles is None:
            self.vehicle_seg()

        for vehi in self.vehicles:
            for i in range(5):
                voxel, pts_centroid, height = vehi.prepare_training_data(
                    self.vehi_reconstructor)
                shape_loss = self.estimate_loss_fun(
                    voxel, pts_centroid, height)
                shape_loss.backward()
                if i == 0:
                    print(shape_loss)
                vehi.vehicle_latent = vehi.vehicle_latent - 1000.0 * vehi.vehicle_latent.grad
                vehi.vehicle_latent = vehi.vehicle_latent.clone().detach()
                vehi.vehicle_latent.requires_grad_(True)

            print(shape_loss)
            # if vehi.translation.grad is not None:
            #     vehi.translation = vehi.translation - vehi.translation.grad
            #     vehi.translation = vehi.translation.clone().detach()
            #     vehi.translation.requires_grad_(True)
            # print(vehi.vehicle_latent.grad)
            # print(vehi.translation.grad)
