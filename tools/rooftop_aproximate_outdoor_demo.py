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

from data_tools import *
from vehicle_reconstruction import *
from scene_segmentation import *
from open3d_vis_utils import draw_scenes

from cudaext.ops.Rotated_IoU.oriented_iou_loss import cal_iou_3d, assign_target_3d

from pytorch3d.vis.plotly_vis import plot_scene

from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

CFG_PATH = "cfgs/dataset_configs/outdoor_demo_dataset.yaml"
INFERENCE_PATH = "/home/ksas/uzuki_space/adv-carla/data/gtound truth/inference_models_outdoor_demo/pointrcnn/gtboxes.pt"
GTBOXES_PATH = "/home/ksas/uzuki_space/adv-carla/data/gtound truth/inference_models_outdoor_demo/pointrcnn/gtboxes.pt"
VISUALIZE = True 
cfg_from_yaml_file(CFG_PATH, cfg)

# BATCH_SIZE = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
logger = common_utils.create_logger()
logger.info('-----------------vehicles reconstruction test-------------------------')

dataset = outdoor_demo_dataset(cfg, 
                                class_names=['Car', 'Pedestrian', 'Cyclist'], 
                                training=False, 
                                ext=".bin", 
                                # inference_path =  INFERENCE_PATH,
                                gtboxes_path = GTBOXES_PATH,
                                logger=logger)

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

for i, batch_dict in tqdm(enumerate(dataset), total=dataset.__len__()):
    if VISUALIZE:
        # batch_dict = dataset.__getitem__(2)
        batch_dict = dataset.__getitem__(random.randrange(0, dataset.__len__()))
        print(f"getting sample (index=: {batch_dict})")
    load_data_to_gpu(batch_dict)
    
    pts = batch_dict['points']
    gt_boxes, gt_labels = torch.split(batch_dict['gt_boxes'], [7, 1], dim=1)  # [N, 7], [N, 1]
    gt_boxes = gt_boxes[gt_labels[:, 0] == 1]
    gt_labels = gt_labels[gt_labels[:, 0] == 1]
    
    if 'inference_bboxes' in batch_dict:
        inference_boxes, inference_labels = torch.split(batch_dict['inference_bboxes'], [7, 1], dim=1)  # [N, 7], [N, 1]
        inference_boxes = inference_boxes[inference_labels[:, 0] == 1]
        inference_labels = inference_labels[inference_labels[:, 0] == 1]
    
    scene = point_cloud_scene(pts=pts,
                          gt_boxes=gt_boxes,
                          vehi_reconstructor=vehi_reconstructor)
    vehicles: list[vehicle_object] = scene.get_vehicles()
    
    scene.pose_estimate(iter=50)
    
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
    
    with open("./rooftop_appro.pkl", "wb") as output:
        pkl.dump(rooftop_array, output)
        
    # with open("./rooftop_appro.pkl", "rb") as input:
    #     print(pkl.load(input))

    if VISUALIZE:

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="view")
        draw_scenes(vis, points=pts, gt_boxes=gt_boxes,
                    draw_origin=True)
    
        for vehi in vehicles:
            if vehi is None:
                continue
            
            mesh, rooftop_idx = vehi.reconstruct_at_scene(
                vehi_reconstructor, standard_bbox, show_rooftop=True)
            
            rooftop_center = mesh.vertices[rooftop_idx].mean(0)
            rooftop_center[2] = mesh.vertices[rooftop_idx][:, 2].max()
            
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.5, origin=rooftop_center)
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
        
        vis.run()
        vis.destroy_window()