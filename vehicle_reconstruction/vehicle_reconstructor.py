import torch
import pytorch3d
from pytorch3d.structures import Meshes
import numpy as np
import json
import skimage
import os

from voxeltorch import TSDF, tsdf2meshes

from tqdm import tqdm

from typing import Union
from collections import OrderedDict


class vehicle(object):

    def __init__(self, name: str = None, vertices: np.ndarray = None, faces: np.ndarray = None, voxel: np.ndarray = None):
        self.name: str = name
        self.vertices: np.ndarray = vertices  # [N, 3]
        self.faces: np.ndarray = faces  # [M, 3]
        self.voxel: np.ndarray = voxel

    def __str__(self) -> str:
        return f"vehicle(\"{self.name}\")"

    def to_mesh(self):
        vehi = None
        if self.vertices is not None and\
                self.faces is not None:
            vehi = vehicle(name=self.name,
                           vertices=self.vertices, faces=self.faces)
        
        elif self.voxel is not None:
            vertices, faces, normals, _ = skimage.measure.marching_cubes(
                self.voxel, gradient_direction='descent', level=0)
            vehi = vehicle(name=self.name,
                           vertices=vertices, faces=faces)
        return vehi
    
    def to_voxel(self, sampling_space, grid_res, truncated_dis=0.2):
        assert self.vertices is not None
        assert self.faces is not None
        voxel = self.voxelize_model(self, sampling_space, grid_res)
        voxel = voxel.clip(-truncated_dis, truncated_dis)
        return vehicle(name=self.name,
                       voxel=voxel)

    def to_trimesh(self):
        mesh = None
        if self.vertices is not None and\
                self.faces is not None:
            mesh = trimesh.Trimesh(
                self.vertices, self.faces)
        elif self.voxel is not None:
            vertices, faces, normals, _ = skimage.measure.marching_cubes(
                self.voxel, gradient_direction='descent', level=0)
            mesh = trimesh.Trimesh(
                vertices, faces)

        return mesh

    def output_as_obj(self, dir_path: str):
        assert self.vertices is not None
        assert self.faces is not None

        with open(dir_path + f"{self.name}.obj", 'w') as f:
            for v in self.vertices:
                f.write("v {} {} {}\n".format(v[0], v[1], v[2]))

            for face in self.faces:
                f.write("f")
                for idx in face:
                    f.write(" {}".format(idx + 1))
                f.write("\n")

    def rooftop_approximate(self, max_dis=2, return_idx=False):
        assert self.vertices is not None
        assert self.faces is not None

        max_y = self.vertices[:, 1].max()
        rooftop_idx = ((max_y - self.vertices[:, 1]) < max_dis)
        rooftop_vertices = self.vertices[rooftop_idx]

        if return_idx:
            return rooftop_vertices, rooftop_idx
        else:
            return rooftop_vertices

    @staticmethod
    def load_car_models_from_obj(car_model_dir: str):
        """Load all the car models
        """
        cars = []
        for filename in os.listdir(car_model_dir):
            if filename.endswith('.obj'):
                file_path = os.path.join(car_model_dir, filename)
                vertices = []
                faces = []
                name = os.path.splitext(filename)[0]
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.startswith('v '):
                            vertex = [float(v)
                                      for v in line.strip().split()[1:]]
                            vertices.append(vertex)
                        elif line.startswith('f '):
                            face = [int(i.split('/')[0]) -
                                    1 for i in line.strip().split()[1:]]
                            faces.append(face)

                vehi = vehicle(name=name,
                               vertices=np.array(vertices),
                               faces=np.array(faces))
                cars.append(vehi)

        return cars

    @staticmethod
    def load_car_models(car_model_dir: str, models):
        """Load all the car models
        """
        cars = []
        print('loading %d car models' % len(models))
        for model in models:
            car_model = '%s/%s.json' % (car_model_dir,
                                        model.name)

            with open(car_model, 'rb') as f:
                data = json.load(f)
                data['vertices'] = np.array(data['vertices'])
                data['vertices'][:, 1] = -data['vertices'][:, 1]
                lwh = np.array(get_model_bbox(data['vertices']))
                data['vertices'][:, 1] -= lwh[5]
                data['faces'] = np.array(data['faces']) - 1
                data['faces'] = data['faces'][:, [2, 1, 0]]

                vehi = vehicle(name=model.name,
                               vertices=np.array(data['vertices']),
                               faces=np.array(data['faces']))
                cars.append(vehi)

        return cars

    @staticmethod
    def voxelize_model(vehi: "vehicle", sampling_space, grid_res):
        bbox = get_model_bbox(vehi.vertices)
        mesh = trimesh.Trimesh(
            vehi.vertices, vehi.faces)
        mesh.process()
        mesh.fill_holes()
        uniform_bbox = (0.0, 0.0, 0.0, -bbox[3]/sampling_space[3], bbox[4]/sampling_space[4],
                        -1.0, bbox[6]/sampling_space[6],
                        -bbox[7]/sampling_space[7], bbox[8]/sampling_space[8])
        voxels: np.ndarray = mesh_to_voxels(mesh, grid_res,
                                            scan_count=100,
                                            bbox=uniform_bbox,
                                            standard_bbox=sampling_space,
                                            scan_resolution=400,
                                            normal_sample_count=200, pad=False)

        return voxels

def quantify_bbox(bbox:torch.Tensor, unit:Union[int, torch.Tensor]):
    quantified_bbox = unit * (bbox / unit).ceil().int()
    return quantified_bbox

class vehicle_reconstructor(object):
    """
        fit and reconstruct vehicle shape via PCA.
    """

    def __init__(self, resolution: Union[int, torch.Tensor], bbox: torch.Tensor, sampling_count=4096,
            downsampling_count=2048, unit:Union[torch.Tensor,float]=None):
        # self.vehicles: Meshes = vehicles
        self.TSDF: TSDF = TSDF(resolution=resolution, sampling_count=sampling_count, downsampling_count=downsampling_count, bbox=bbox)
        self.unit = unit
        
        self._U = None
        self._S = None
        self._V = None
        
    def save_parameters(self, path:str):
        assert (self._U is not None) and (self._S is not None) and (self._V is not None) and (self.batch_tsdf_mean is not None)
        torch.save((self._U, self._S, self._V, self.batch_tsdf_mean), path)
    
    def load_parameters(self, path:str):
        self._U, self._S, self._V, self.batch_tsdf_mean = torch.load(path)

    def prepare_tsdf(self, vehicles: Meshes):
        """
            Computing tsdf grid of the Meshes
            Args:
                vehicles: Meshes
            
            Returns:
                tsdf_grid: [B, l, w, h]

        """
        batch_tsdf_grid:torch.Tensor = self.TSDF.tsdf(vehicles)

        return batch_tsdf_grid

    def fit_meshes(self, vehicles: Meshes, k=5):
        """
            Fitting the Meshes with PCA
            Args:
                vehicles: Meshes
                k: latent dimension
            
        """
        self.batch_tsdf_grid = self.prepare_tsdf(vehicles)

        batch_tsdf_flatten = self.batch_tsdf_grid.view(self.batch_tsdf_grid.size(0), -1)
        
        self.batch_tsdf_mean = batch_tsdf_flatten.mean(dim = 0)
        self._U, self._S, self._V = torch.pca_lowrank(batch_tsdf_flatten - self.batch_tsdf_mean, q=k, center=True)
        
    def reconsturct(self, vehicles: Union[Meshes, torch.Tensor]):
        if isinstance(vehicles, torch.Tensor):
            latent = self.encode_aux(vehicles.view(vehicles.size(0), -1))
        elif isinstance(vehicles, Meshes):
            latent = self.encode(vehicles)

        return self.decode(latent)

    def encode(self, vehicles: Meshes):
        batch_tsdf_grid = self.prepare_tsdf(vehicles)
        batch_tsdf_flatten = batch_tsdf_grid.view(batch_tsdf_grid.size(0), -1)
        
        return self.encode_aux(batch_tsdf_flatten)

    def encode_aux(self, batch_tsdf):
        assert self._V is not None
        latent = (batch_tsdf - self.batch_tsdf_mean) @ self._V @ self._S.diag().inverse()

        return latent

    def decode(self, latent:torch.Tensor, to_meshes:bool=False):
        batch_reconstructed_tsdf = self.decode_aux(latent).view(-1, *self.TSDF.resolution)
        return batch_reconstructed_tsdf

    def decode_aux(self, latent):
        assert self._V is not None
        batch_reconstructed_tsdf = latent @ self._S.diag() @ self._V.T + self.batch_tsdf_mean

        return batch_reconstructed_tsdf


if __name__ == "__main__":
    pass
