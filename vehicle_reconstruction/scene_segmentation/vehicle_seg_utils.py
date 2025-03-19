# from __future__ import annotations

import numpy as np
from pytorch3d import transforms
from pytorch3d.ops.mesh_filtering import taubin_smoothing
from pytorch3d.structures import Pointclouds
import torch
import torch.nn.functional as F

from vehicle_reconstruction import *
from cudaext.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from cudaext.ops.roipoint_pool3d.roipoint_pool3d_utils import RoIPointPool3d
from vehicle_reconstruction.scene_segmentation.loss_utils import shape_regular, pose_estimate_loss_batch
from voxeltorch import TSDF, tsdf2meshes

class vehicle_object:
    def __init__(self, pts: Pointclouds, bbox: torch.Tensor, training_feature:torch.Tensor=None, k=5) -> None:
        """
            pts: [B, N, 3]
            bbox: [B, 7]
        """
        self.pts =  pts
        self.bbox = bbox.detach().clone() # [x, y, z, l, w, h, cos]
        self.batch_size = pts.__len__()
        # self.vehicle: "vehicle" = None

        self.training_feature = training_feature
        self.vehicle_latent = torch.zeros(
            (self.batch_size, k), requires_grad=True, device=pts.device)
        # [1, K]

        self.translation = torch.stack([bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 6]], dim = 1)
        self.translation.requires_grad_(True)

        self._local_world_translate_m = transforms.Translate(self.bbox[:, :3])
        self._local_world_rotate_m = transforms.RotateAxisAngle(self.bbox[:, 6], axis="Z", degrees=False)
        self._local_world_centering_heights = transforms.Translate(-torch.concat([self.bbox.new_zeros((self.bbox.size(0), 2)), 
                                                                      self.bbox[:, 5:6]], dim=1).to(self.pts.device) / 2)
        
        self._world_local_translate_m = transforms.Translate(-self.bbox[:, :3])
        self._world_local_rotate_m = transforms.RotateAxisAngle(-self.bbox[:, 6], axis="Z", degrees=False)
        self._world_local_centering_heights = transforms.Translate(torch.concat([self.bbox.new_zeros((self.bbox.size(0), 2)), 
                                                                      self.bbox[:, 5:6]], dim=1).to(self.pts.device) / 2)

        self._local_positive_centering = transforms.Translate(torch.concat([self.bbox[:, 3:5],
                                                            self.bbox.new_zeros(self.bbox.size(0), 1)], dim=1).to(self.pts.device) / 2)

    def __len__(self):
        return self.pts.__len__()

    def reconstruct_at_scene(self, vehi_reconstructor:vehicle_reconstructor, 
                             threshold:int=10,
                             smoothing:bool=True,
                             show_rooftop: bool = False):
        
        threshold_flag = (self.pts.num_points_per_cloud() > threshold)
        packed_rooftop_idx = None
        with torch.no_grad():
            tsdf_grid = vehi_reconstructor.decode(self.vehicle_latent)
            vehi = tsdf2meshes(tsdf_grid, vehi_reconstructor.unit, padding=1) # Add padding to avoid non-watertight meshes
            vehi.offset_verts_(-vehi_reconstructor.unit) # eliminate the offset by padding

            centering_bbox = -torch.Tensor([vehi_reconstructor.TSDF.bbox[0], 0.0, vehi_reconstructor.TSDF.bbox[2]]).to(vehi.device) / 2
            vehi.offset_verts_(centering_bbox) # centering the meshes

            vehi = vehi.update_padded(vehi.verts_padded()[:, :, [2, 0, 1]]) # [x, y, z] -> [z, x, y]
            vehi = vehi.update_padded(self._local_world_rotate_m.transform_points(vehi.verts_padded())) # local rotation
            vehi = vehi.update_padded(self._local_world_centering_heights.transform_points(vehi.verts_padded()))
            vehi = vehi.update_padded(self._local_world_translate_m.transform_points(vehi.verts_padded()))
            vehi = vehi[threshold_flag]

            if smoothing:
                vehi = taubin_smoothing(vehi)
            
            if show_rooftop:
                # max_z = vehi.verts_padded()[:, :, 2].amax(dim = 1, keepdim=True)
                positive_vehi = vehi.offset_verts(torch.maximum(-vehi.verts_padded().amin(dim=(0, 1)), centering_bbox.new_ones(3)))
                max_z = positive_vehi.verts_padded()[:, :, 2].amax(dim = 1, keepdim=True)
                packed_rooftop_idx:torch.Tensor = (max_z - positive_vehi.verts_padded()[:, :, 2]) < 0.2
                packed_rooftop_idx = packed_rooftop_idx.view(-1)[vehi.verts_padded_to_packed_idx()]
                
        
        return vehi, packed_rooftop_idx
    
    def get_local_point_clouds(self, pts:Pointclouds=None):
        """
            World coordinate -> local centroid coordinate (positive Z)
        """
        centroid_pts = pts or self.pts
        centroid_pts = centroid_pts.update_padded(self._world_local_translate_m.transform_points(centroid_pts.points_padded()))
        centroid_pts = centroid_pts.update_padded(self._world_local_centering_heights.transform_points(centroid_pts.points_padded()))
        centroid_pts  = centroid_pts.update_padded(self._world_local_rotate_m.transform_points(centroid_pts.points_padded()))
        
        return centroid_pts

    def get_training_point_clouds(self):
        """
            World coordinate -> local positive coordinate
        """
        local_pts = self.get_local_point_clouds(Pointclouds(self.training_feature))
        postive_pts  = local_pts.update_padded(self._local_positive_centering.transform_points(local_pts.points_padded()))
        postive_pts.update_padded(postive_pts.points_padded()[:, :, [1, 2, 0]]) # [z, x, y] -> [x, y, z]
        return postive_pts
    
    def get_tsdf_grid(self, vehi_reconstructor:vehicle_reconstructor):
        """
            Get tsdf from latent
        """
        tsdf_grid = vehi_reconstructor.decode(self.vehicle_latent)
        return tsdf_grid
    


class point_cloud_scene:

    def __init__(self, pts: torch.Tensor, gt_boxes: torch.Tensor) -> None:
        """
            pts: [N, 4]
            gt_boxes: [N, 7]
        """
        self.pts: torch.Tensor = pts.detach().clone()
        self.gt_boxes: torch.Tensor = gt_boxes.detach().clone()
        self.vehicles: vehicle_object = None
        
        self.pose_estimate_loss_batch = pose_estimate_loss_batch()
        self.shape_regular_fun: shape_regular = shape_regular()

        self.roipool3d = RoIPointPool3d(pool_extra_width = (0, 0, 0))

        self.segmented_pts = None # [B, N, 3]
        self.segmented_gt_boxes = None #[B, 7]



    
    def vehicle_seg(self):
        self.pts_assign = points_in_boxes_gpu(
            self.pts[:, :3].view(1, -1, 3), self.gt_boxes.view(1, -1, 7)).squeeze(dim=0)
        
        self.segmented_pts = []
        self.segmented_gt_boxes = []

        for bbox_idx in range(self.gt_boxes.size(0)):
            bbox_mask = (self.pts_assign == bbox_idx)
            self.segmented_pts.append(self.pts[bbox_mask, :3])
            self.segmented_gt_boxes.append(self.gt_boxes[bbox_idx])

        training_feature, pooled_empty_flag = self.roipool3d.forward(self.pts[:, :3].view(1, -1, 3),
                                                            self.pts.new_empty((1, self.pts.size(1), 0)),
                                                            self.gt_boxes.view(1, -1, 7))
        training_feature:torch.Tensor = training_feature.squeeze(0)
        
        self.segmented_pts = Pointclouds(self.segmented_pts)
        self.segmented_gt_boxes = torch.stack(self.segmented_gt_boxes)
        self.vehicles = vehicle_object(self.segmented_pts, self.segmented_gt_boxes, training_feature)

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


    def pose_estimate(self, vehi_reconstructor:vehicle_reconstructor, iter=5, point_threshold:int=10):
        if self.vehicles is None:
            self.vehicle_seg()

        import time
        start_time = time.time()
        for _ in range(iter):
            threshold_flag = (self.vehicles.pts.num_points_per_cloud() > point_threshold)
            training_point_clouds = self.vehicles.get_training_point_clouds()
            tsdf_grid = self.vehicles.get_tsdf_grid(vehi_reconstructor)

            training_point_clouds = training_point_clouds[threshold_flag]
            tsdf_grid = tsdf_grid[threshold_flag]

            estimate_loss = self.pose_estimate_loss_batch.forward(tsdf_grid, training_point_clouds.points_padded(), vehi_reconstructor.unit)
            shape_regulation_loss = self.shape_regular_fun.forward(
                    self.vehicles.vehicle_latent[threshold_flag], vehi_reconstructor._S)
            totall_loss = estimate_loss + shape_regulation_loss
            totall_loss.backward()

            grad = self.vehicles.vehicle_latent.grad
            self.vehicles.vehicle_latent = self.vehicles.vehicle_latent - grad
            self.vehicles.vehicle_latent = self.vehicles.vehicle_latent.clone().detach()
            self.vehicles.vehicle_latent.requires_grad_(True)
        end_time = time.time()
        print(f"Time cost:{end_time - start_time}, {(end_time - start_time)/iter} per epoch")
