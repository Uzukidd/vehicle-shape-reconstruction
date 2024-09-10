import torch
import numpy as np
import json
import trimesh
import skimage
import os

from tqdm import tqdm
# from mesh_to_sdf import mesh_to_voxels

from collections import OrderedDict
from .utils import get_model_bbox


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


class vehicle_reconstructor(object):
    """
        fit and reconstruct vehicle shape via PCA.
    """
    GLOBAL_RES:float = 0.1
    STANDARD_BBOX = [2.5, 1.9, 5.6, -1.25, 1.25, 0.0, 1.90, -2.8, 2.8]
    GRID_RES = [int(np.ceil(STANDARD_BBOX[0]/GLOBAL_RES) + 1),
                int(np.ceil(STANDARD_BBOX[1]/GLOBAL_RES) + 1),
                int(np.ceil(STANDARD_BBOX[2]/GLOBAL_RES) + 1)]
    
    def __init__(self, vehicles: list[vehicle], sampling_space, grid_res, global_res):
        self.vehicles: list[vehicle] = vehicles
        self.vehicle_voxels: np.ndarray = None  # [N, res1, res2, res3]
        self.average_voxels: torch.Tensor = None  # [N, K]
        self.sampling_space = sampling_space
        self.grid_res = grid_res
        self.global_res = global_res

        self.U = None
        self.S = None
        self.V = None
        
    def save_parameters(self, path:str):
        assert (self.U is not None) and (self.S is not None) and (self.V is not None)
        torch.save((self.U, self.S, self.V), path)
    
    def load_parameters(self, path:str):
        self.U, self.S, self.V = torch.load(path)

    def save_voxel(self, path: str):
        assert self.vehicle_voxels is not None
        self.vehicle_voxels.astype(np.float32).tofile(path)

    def load_voxel(self, path: str):
        self.vehicle_voxels = np.fromfile(
            path, np.float32).reshape([-1] + self.grid_res)

    def voxelize_vehicles(self):
        self.vehicle_voxels = []

        for vehi in tqdm(self.vehicles):
            voxel = vehi.to_voxel(self.sampling_space, self.grid_res).voxel
            self.vehicle_voxels.append(voxel)

        self.vehicle_voxels = np.stack(self.vehicle_voxels)

    def fit_model(self, k=4):
        assert self.vehicle_voxels is not None
        stack_voxels = torch.from_numpy(
            self.vehicle_voxels).cuda().view(self.vehicle_voxels.__len__(), -1)

        self.average_voxels = stack_voxels.mean(dim=0)
        self.U, self.S, self.V = torch.pca_lowrank(
            stack_voxels, q=k, center=True)

    def reconsturct(self, vehicles: list[vehicle] = None):
        if vehicles is not None:
            stack_latent = self.encode(vehicles)
        else:
            stack_latent = self.encode_aux(self.vehicle_voxels)

        return self.decode(stack_latent)

    def encode(self, vehicles: list[vehicle]):
        voxels = []
        for vehi in vehicles:
            voxel = vehi.voxel
            if vehi.voxel is None:
                voxel = vehi.to_voxel(self.sampling_space, self.grid_res).voxel

            voxels.append(voxel)

        voxels = np.stack(voxels)

        return self.encode_aux(voxels)

    def encode_aux(self, voxels):
        assert self.V is not None

        voxels = torch.from_numpy(
            voxels).cuda().view(voxels.shape[0], -1) - self.average_voxels

        stack_latent = torch.matmul(voxels, self.V)

        return stack_latent

    def decode(self, latent):
        voxels = self.decode_aux(latent).detach().cpu().numpy()
        vehicles = []
        for voxel in voxels:
            vehicles.append(vehicle(voxel=voxel.reshape(self.grid_res)))

        return vehicles

    def decode_aux(self, latent):
        assert self.V is not None

        reconstructed_voxel = torch.matmul(
            latent, self.V.T) + self.average_voxels
        reconstructed_voxel = reconstructed_voxel.view([-1] + self.grid_res)
        return reconstructed_voxel


if __name__ == "__main__":
    import car_models
    CAR_MODEL_DIR = "F:\\ApolloScape\\3d_car_instance_sample\\3d_car_instance_sample\\car_models_json"
    vehicles = vehicle.load_car_models(car_model_dir=CAR_MODEL_DIR,
                                       models=car_models.models)
