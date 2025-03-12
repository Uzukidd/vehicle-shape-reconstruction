import os

import torch
from torch.utils.data import dataset, dataloader
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from pprint import pprint

from .constants import apolloscape_constant


class apolloscape_dataset():

    def __init__(self, data_root:str = None, car_model_summary:list = None):
        self.car_model_summary = car_model_summary or apolloscape_constant.models
        self.data_root = data_root or apolloscape_constant._apolloscape_DIR

        self.prepare_input()

    def prepare_input(self):
        self.apolloscape_meshes = __class__.load_car_models(self.data_root, self.car_model_summary)

    @staticmethod
    def load_car_models(data_root: str, models:list):
        """Load all the car models
        """
        print('loading %d car models' % len(models))
        apolloscape_meshes = None
        verts_list, faces_list = list(), list()
        for model in models:
            car_model_path = os.path.join(data_root, f"{model.name}.obj")
            verts, faces, _ = load_obj(car_model_path, load_textures=False)
            verts_list.append(verts)
            faces_list.append(faces.verts_idx)

            apolloscape_meshes = Meshes(verts_list, faces_list)
        
        return apolloscape_meshes
    
    def get_batch_centered_meshes(self):
        bbox = self.get_bbox()
        apolloscape_meshes = self.get_batch_meshes()
        # Move all the vehicle to the center
        apolloscape_meshes.offset_verts_(torch.Tensor([0.0, -bbox[1].item()/2, 0.0]).to(apolloscape_meshes.device))

        return apolloscape_meshes

    def get_batch_meshes(self):
        return self.apolloscape_meshes
    
    def get_bbox(self):
        bbox = self.apolloscape_meshes.verts_packed().amax(
            0) - self.apolloscape_meshes.verts_packed().amin(0)
        
        return bbox


    
if __name__ == "__main__":
    apolloscape_constant.set_apolloscape_dir("/mnt/sda/uzuki_space/voxeltorch/assets/apollo_scape")
    apolloscape = apolloscape_dataset()
    apolloscape_meshes = apolloscape.get_batch_meshes()
    print(apolloscape_meshes.__len__())
    print(apolloscape.get_bbox())
    print(apolloscape.get_batch_centered_meshes())
