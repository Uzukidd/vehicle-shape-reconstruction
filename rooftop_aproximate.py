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

import pdb
import open3d as o3d
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
from pathlib import Path

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

EVAL_OUTPUT_DIR = "./eval_output/"
CFG_FILE = "./cfgs/kitti_models/pointrcnn.yaml"
DATA_CONFIG_FILE = "./cfgs/dataset_configs/kitti_dataset.yaml"
DATA_PATH = "/home/ksas/Public/datasets/KITTI"
CKPT_PATH = "/home/ksas/Public/model_zoo/pcdet/pointrcnn_7870.pth"

BATCH_SIZE = 1
WORKERS = 4
DIST_TEST = False

VISUALIZE = False

OPTIM = "rbboxloss"
EVAL_INIT_PATH = False

cfg_from_yaml_file(CFG_FILE, cfg)

# BATCH_SIZE = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
logger = common_utils.create_logger()
logger.info('-----------------vehicles reconstruction test-------------------------')

laplacian_weights = 0.001
learning_rate = 0.005
overshoot = 0.02

test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=BATCH_SIZE,
        dist=DIST_TEST, workers=WORKERS, logger=logger, training=False
    )
logger.info(f'Class names of samples: \t{test_set.class_names}')

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

for i, batch_dict in tqdm(enumerate(test_set), total=test_set.__len__()):
    # batch_dict = test_set.__getitem__(13)
    load_data_to_gpu(batch_dict)
    
    pts = batch_dict['points']
    gt_boxes, gt_labels = torch.split(batch_dict['gt_boxes'], [7, 1], dim=1)  # [N, 7], [N, 1]
    gt_boxes = gt_boxes[gt_labels[:, 0] == 1]
    gt_labels = gt_labels[gt_labels[:, 0] == 1]
    
    scene = point_cloud_scene(pts=pts,
                          gt_boxes=gt_boxes,
                          vehi_reconstructor=vehi_reconstructor)
    vehicles: list[vehicle_object] = scene.get_vehicles()
    # print(vehicles.__len__())
    
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
                    size=1.0, origin=rooftop_center)
            vis.add_geometry(axis_pcd)
            
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh.vertices),
                o3d.utility.Vector3iVector(mesh.faces))
            

            vertex_colors = np.ones_like(mesh.vertices) * np.array([1.0, 0.5, 0.5])
            vertex_colors[rooftop_idx] = np.array([0.5, 1.0, 0.5])
            vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            mesh.vertex_colors = vertex_colors
            # mesh = mesh.filter_smooth_simple(number_of_iterations=5)
            mesh.compute_vertex_normals()

            vis.add_geometry(mesh)
        
        vis.run()
        vis.destroy_window()