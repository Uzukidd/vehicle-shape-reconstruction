UNI_RANDOM_SEED = 2024
DEVICE = 1

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(UNI_RANDOM_SEED) 
torch.manual_seed(UNI_RANDOM_SEED)

torch.cuda.manual_seed(UNI_RANDOM_SEED)
torch.cuda.manual_seed_all(UNI_RANDOM_SEED)

torch.cuda.set_device(DEVICE)

import random
import pdb
import open3d as o3d
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from data_tools import *
from vehicle_reconstruction import *
from vehicle_reconstruction.scene_segmentation import *
from open3d_vis_utils import draw_scenes

from cudaext.ops.Rotated_IoU.oriented_iou_loss import cal_iou_3d, assign_target_3d

from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import plot_scene

from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from attack_utils import *

CFG_PATH = "cfgs/dataset_configs/outdoor_demo_dataset.yaml"

ADV_PATCH_PATH = "/home/ksas/uzuki_space/adv-carla/output/train/loss_ablation/pointrcnn/misrecognize_10/initial_patch_checkpoint.pt"
GTBOXES_PATH = "/home/ksas/uzuki_space/adv-carla/output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt"

PRE_SCORE_PATH = "/home/ksas/uzuki_space/adv-carla/output/train/inference_models_outdoor_demo/pointrcnn/pre_scores.pt"
ADV_PRE_SCORE_PATH = "/home/ksas/uzuki_space/adv-carla/output/train/inference_models_outdoor_demo/pointrcnn_adv/adversarial_pre_scores.pt"
ADVERSARIAL_BBOXES_PATH = "/home/ksas/uzuki_space/adv-carla/output/train/inference_models_outdoor_demo/pointrcnn_adv/adversarial_boxes.pt"

VISUALIZE = True 
cfg_from_yaml_file(CFG_PATH, cfg)

# BATCH_SIZE = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
logger = common_utils.create_logger()
logger.info('-----------------vehicles reconstruction test-------------------------')

# dataset = outdoor_demo_dataset(cfg, 
#                                 class_names=['Car', 'Pedestrian', 'Cyclist'], 
#                                 training=False, 
#                                 ext=".bin", 
#                                 # inference_path =  INFERENCE_PATH,
#                                 gtboxes_path = GTBOXES_PATH,
#                                 logger=logger)

global_res = 0.1
standard_bbox = [2.5, 1.9, 5.6, -1.25, 1.25, 0.0, 1.90, -2.8, 2.8]
grid_res = [int(np.ceil(standard_bbox[0]/global_res) + 1),
            int(np.ceil(standard_bbox[1]/global_res) + 1),
            int(np.ceil(standard_bbox[2]/global_res) + 1)]
print(grid_res)
vehi_reconstructor = vehicle_reconstructor(vehicles=None,
                                           sampling_space=standard_bbox,
                                           grid_res=grid_res,
                                           global_res=global_res)
vehi_reconstructor.load_voxel("./data/preprocessed_voxel.bin")
vehi_reconstructor.fit_model(k=5)

rooftop_array = []

adv_predicted_scores = None
if ADV_PRE_SCORE_PATH is not None:
    adv_predicted_scores = torch.load(ADV_PRE_SCORE_PATH)
    
predicted_scores = None
if PRE_SCORE_PATH is not None:
    predicted_scores = torch.load(PRE_SCORE_PATH)
    
adversarial_bboxes = None
if ADVERSARIAL_BBOXES_PATH is not None:
    adversarial_bboxes = torch.load(ADVERSARIAL_BBOXES_PATH)

adv_patches = None
if ADV_PATCH_PATH is not None:
    adv_patches_params = torch.load(ADV_PATCH_PATH)["universal_adv_patch_car"]
    adv_patches = single_sphere()
    adv_patches.load_parameter(adv_patches_params)

# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 19.432529449462891, 39.944240570068359, 9.9070224761962891 ],
# 			"boundingbox_min" : [ -0.059999999999999998, -38.348209381103516, -1.4829977750778198 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.80890813089305436, 0.27625877374859964, 0.51898817491540739 ],
# 			"lookat" : [ 3.8872254470312524, 8.4100763521246371, 2.932072239929659 ],
# 			"up" : [ 0.56714512088524349, 0.13398401833700907, 0.81264672194400112 ],
# 			"zoom" : 0.079999999999999613
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }

def draw_reconstructed_scene(pts:torch.Tensor, 
                             lidar:Optional[LiDAR_base] = None,
                             gt_boxes:Optional[torch.Tensor] = None, 
                             ref_boxes:Optional[torch.Tensor] = None, 
                             vehicles:Optional[list[vehicle_object]] = None,
                             adv_patch:Optional[adversarial_patch_3d] = None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="view")
    meshes_batch = []
    pts_set = [pts]
    if vehicles is not None:
        for vehi in vehicles:
            if vehi is None:
                continue
            
            mesh, rooftop_idx = vehi.reconstruct_at_scene(
                vehi_reconstructor, standard_bbox, show_rooftop=True)
            
            rooftop_center = mesh.vertices[rooftop_idx].mean(0)
            rooftop_center[2] = mesh.vertices[rooftop_idx][:, 2].max()
            
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.5, origin=rooftop_center)
            
            if lidar is None:
                vis.add_geometry(axis_pcd)
                
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(mesh.vertices),
                    o3d.utility.Vector3iVector(mesh.faces))

                vertex_colors = np.ones_like(mesh.vertices) * np.array([0.5, 1.0, 0.5])
                vertex_colors[rooftop_idx] = np.array([1.0, 0.5, 0.5])
                vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                mesh.vertex_colors = vertex_colors
                # mesh = mesh.filter_smooth_simple(number_of_iterations=5)
                mesh.compute_vertex_normals()

                vis.add_geometry(mesh)
                
            if adv_patch is not None:
                transformed_mesh:Meshes = adv_patch.get_transformed_meshes(torch.from_numpy(rooftop_center).float().cuda(),
                                                                            vehi.translation[3].view((1)))
                if lidar is not None:
                    meshes_batch.append(transformed_mesh)
                else:
                    mesh = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(transformed_mesh.verts_packed().detach().cpu().numpy()),
                        o3d.utility.Vector3iVector(transformed_mesh.faces_packed().detach().cpu().numpy()))

                    mesh.compute_vertex_normals()
                
                    vis.add_geometry(mesh)
    
    point_color = None
    if meshes_batch.__len__() != 0:
        meshes_batch = join_meshes_as_batch(meshes_batch)
        extend_pts = lidar.scan_triangles(meshes_batch)
        extend_pts = F.pad(extend_pts,  (0, 1), "constant", 0)
        point_color = torch.concatenate([torch.ones_like(pts_set[0])[:, :3].detach().cpu(), torch.ones_like(extend_pts)[:, :3].detach().cpu() * torch.tensor([1.0, 0.0, 0.0])])
        pts_set.append(extend_pts)
    
    ref_labels = None
    if ref_boxes is not None:
        ref_labels = ref_boxes.new_ones((ref_boxes.size(0))).long()
    draw_scenes(vis, points=torch.concatenate(pts_set), gt_boxes=gt_boxes,
                point_colors = point_color,
                ref_labels=ref_labels,
                ref_boxes=ref_boxes,
            draw_origin=True)
    vis.run()
    vis.destroy_window()

def rotate_points(points: torch.Tensor, angle: torch.Tensor):
    """
    Rotate a set of points around the origin by a given angle.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing N points in 3D space.
        angle (torch.Tensor): The angle of rotation in radians.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) containing the rotated points.
    """
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    rotation_matrix = points.new_tensor([
        [cos_theta, -sin_theta, 0.0],
        [sin_theta, cos_theta, 0.0],
        [0.0, 0.0, 1.0]
    ])

    rotated_points = torch.matmul(points, rotation_matrix.T)
    return rotated_points

def attach_adv_patch_scene_car_aux(lidar:LiDAR_base, points, adv_patch:adversarial_patch_3d, 
                            theta, 
                            rooftop_approximate: np.ndarray = None,
                            sample_amount = 50,
                            adversarial_parameters = None,):
    pts_set = [points]
    meshes_batch = []
    if rooftop_approximate is not None:
        n = rooftop_approximate.shape[0]
        for i in range(n):
            extend_pts = None
            if lidar is not None:
                transformed_mesh = adv_patch.get_transformed_meshes(torch.from_numpy(rooftop_approximate[i]).float().cuda(),
                                                                        theta[i],
                                                                        adversarial_parameters)
                meshes_batch.append(transformed_mesh)
                
        if meshes_batch.__len__() != 0:
            meshes_batch = join_meshes_as_batch(meshes_batch)
            extend_pts = lidar.scan_triangles(meshes_batch)
            extend_pts = F.pad(extend_pts,  (0, 1), "constant", 0)
            pts_set.append(extend_pts)
            
    return torch.concatenate(pts_set)

dataset = outdoor_demo_dataset(cfg, 
                                class_names=['Car', 'Pedestrian', 'Cyclist'], 
                                training=False, 
                                ext=".bin", 
                                # inference_path =  INFERENCE_PATH,
                                gtboxes_path = GTBOXES_PATH,
                                logger=logger)

for i, batch_dict in tqdm(enumerate(dataset)):
    index = i
    if VISUALIZE:
        # batch_dict = dataset.__getitem__(2)
        # index = 44
        # index = 18
        pred_scores = torch.sigmoid(predicted_scores[index])
        batch_dict = dataset.__getitem__(index)
        print(f"getting sample (index=: {index})")
        
    pred_scores =  torch.sigmoid(predicted_scores[index])
    adv_pred_scores =  torch.sigmoid(adv_predicted_scores[index])
    adv_boxes = adversarial_bboxes[index]
    load_data_to_gpu(batch_dict)
    print(pred_scores)
    print(adv_pred_scores)
    pts = batch_dict['points']
    gt_boxes, gt_labels = torch.split(batch_dict['gt_boxes'], [7, 1], dim=1)  # [N, 7], [N, 1]
    gt_boxes = gt_boxes[gt_labels[:, 0] == 1]
    gt_labels = gt_labels[gt_labels[:, 0] == 1]
    

    scene = point_cloud_scene(pts=pts,
                          gt_boxes=gt_boxes,
                          vehi_reconstructor=vehi_reconstructor)
    vehicles: list[vehicle_object] = scene.get_vehicles()
    lidar = LiDAR_base(origin=torch.tensor([0.0, 0.0, 0.0]).cuda(),
                azi_range=[-90, 90],
                polar_range= [-24.8, 2.0],
                polar_num=10, azi_res=0.08)


    scene.pose_estimate(iter=20)
    
    vehi_rooftop = []
    
    for vehi in vehicles:
        if vehi is None:
            vehi_rooftop.append(None)
            continue
        mesh, rooftop_idx = vehi.reconstruct_at_scene(
            vehi_reconstructor, standard_bbox, show_rooftop=True)
        rooftop_center = mesh.vertices[rooftop_idx].mean(0)
        rooftop_center[2] = mesh.vertices[rooftop_idx][:, 2].max()
        rooftop_center = np.array(rooftop_center)
        vehi_rooftop.append(rooftop_center)
    
    rooftop_array.append(vehi_rooftop)
    if not VISUALIZE:
        with open("./rooftop_appro.pkl", "wb") as output:
            pkl.dump(rooftop_array, output)

    if VISUALIZE:
        draw_reconstructed_scene(pts=pts,
                                 lidar=None,
                                 gt_boxes=gt_boxes,
                                 vehicles=None,
                                 adv_patch=None)
        draw_reconstructed_scene(pts=pts,
                                gt_boxes=gt_boxes,
                                # ref_boxes=adv_boxes,
                                vehicles=vehicles,
                                adv_patch=None)
        draw_reconstructed_scene(pts=pts,
                                 gt_boxes=gt_boxes,
                                 vehicles=vehicles,
                                #  ref_boxes=adv_boxes,
                                 adv_patch=adv_patches)
        # draw_reconstructed_scene(pts=pts,
        #                         #  gt_boxes=gt_boxes,
        #                          vehicles=vehicles,
        #                         #  ref_boxes=adv_boxes,
        #                          adv_patch=adv_patches)
        draw_reconstructed_scene(pts=pts,
                                 lidar=lidar,
                                #  gt_boxes=gt_boxes,
                                #  ref_boxes=adv_boxes,
                                 vehicles=vehicles,
                                 adv_patch=adv_patches)
        draw_reconstructed_scene(pts=pts,
                            lidar=lidar,
                        #  gt_boxes=gt_boxes,
                            ref_boxes=adv_boxes,
                            vehicles=vehicles,
                            adv_patch=adv_patches)
        """
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 64.48394775390625, 39.969989776611328, 13.450284004211426 ],
			"boundingbox_min" : [ -0.07192191857652741, -39.883628845214844, -2.0746853351593018 ],
			"field_of_view" : 60.0,
			"front" : [ -0.62256980582239485, 0.34196126210144068, 0.70389582474983869 ],
			"lookat" : [ 12.533068624233101, -7.2164355359373298, -11.703165472214195 ],
			"up" : [ 0.37894192389910542, -0.65525157727863048, 0.65348939454708188 ],
			"zoom" : 0.23999999999999957
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}"""
        
        """
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 48.698535919189453, 39.966701507568359, 13.827677726745605 ],
			"boundingbox_min" : [ -0.059999999999999998, -39.935523986816406, -2.0649197101593018 ],
			"field_of_view" : 60.0,
			"front" : [ -0.55615931574019251, -0.82078744333331533, 0.13036406092846586 ],
			"lookat" : [ 0.48666435732690383, -5.6615505405449866, 3.7143171587939823 ],
			"up" : [ -0.14755675140481847, 0.25189278394903003, 0.95643976836456568 ],
			"zoom" : 0.19999999999999998
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
"""