import torch
import numpy as np

import pytorch3d
from pytorch3d.ops import sample_points_from_meshes, laplacian
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from pytorch3d.transforms import euler_angles_to_matrix

from abc import ABC, abstractmethod

class learnable_cube:
    CUBE_VERTEICE = torch.tensor([
        [-1, -1, -1],  # 0
        [-1, -1,  1],  # 1
        [-1,  1, -1],  # 2
        [-1,  1,  1],  # 3
        [ 1, -1, -1],  # 4
        [ 1, -1,  1],  # 5
        [ 1,  1, -1],  # 6
        [ 1,  1,  1]   # 7
    ], dtype=torch.float32)

    # 定义立方体的面（由顶点索引组成的三角形）
    CUBE_FACES = torch.tensor([
        [0, 1, 2], [1, 3, 2],  # 左面
        [4, 6, 5], [5, 6, 7],  # 右面
        [0, 4, 1], [1, 4, 5],  # 底面
        [2, 3, 6], [3, 7, 6],  # 顶面
        [0, 2, 4], [2, 6, 4],  # 后面
        [1, 5, 3], [3, 5, 7]   # 前面
    ], dtype=torch.int64)
    def __init__(self) -> None:
        pass

class learnable_sphere:
    
    def __init__(self, level:int = 2,
                    scale:list=[0.35, 0.35, 0.25],
                    eps = 0.0):
        self.scale: torch.Tensor = torch.tensor(scale).float().cuda()
        self.basic_mesh = self.generate_basic_mesh(level=level, 
                                                   scale=scale, 
                                                   eps=eps)
        
        self.deform_vert_logit: torch.Tensor = torch.zeros_like(self.basic_mesh.verts_packed(), requires_grad=True).cuda().contiguous()
        self.deform_vert_logit.requires_grad_(True)
        
        self.init_vert_quadrant: torch.Tensor = torch.sign(self.basic_mesh.verts_packed()).detach().clone()
        self.init_vert_logit: torch.Tensor = torch.logit(torch.abs(self.basic_mesh.verts_packed() / 
                                                                   self.scale[None, :])).detach().clone()
        print(f"mesh vertex count : {self.basic_mesh.verts_packed().size()}")

    
    def get_parameters(self) -> list[torch.Tensor]:
        return [self.deform_vert_logit]
    
    def get_basic_meshes(self) -> Meshes:
        return self.basic_mesh
    
    def get_deformed_meshes(self, deform_vert_logit:torch.Tensor = None) -> Meshes:
        basic_mesh = self.get_basic_meshes()
        if deform_vert_logit is None:
            deform_vert_logit = self.deform_vert_logit
            
        deformed_vert = self.scale[None, :] \
                * self.init_vert_quadrant \
                * torch.sigmoid(self.init_vert_logit + deform_vert_logit)

        return basic_mesh.update_padded(deformed_vert.unsqueeze(0))
    
    def get_transformed_meshes(self, translate:torch.Tensor, deform_vert_logit:torch.Tensor = None):
        deformed_meshes = self.get_deformed_meshes(deform_vert_logit = deform_vert_logit)
        verts = deformed_meshes.verts_padded()
        verts = verts + translate[None, :]
        
        return deformed_meshes.update_padded(verts)
    
    def get_base_coord(self):
        base_z = self.basic_mesh.verts_packed()[:, 2].min()
        
        return self.deform_vert.new_tensor([0.0, 0.0, base_z], requires_grad=False)
    
    @staticmethod
    def generate_basic_mesh(level:int,
                            scale:list,
                            eps):
        mSphere = ico_sphere(level).cuda()

        new_vert = mSphere.verts_padded()
        new_vert[:, :, 0] = new_vert[:, :, 0] * scale[0] * (1 - eps)
        new_vert[:, :, 1] = new_vert[:, :, 1] * scale[1] * (1 - eps)
        new_vert[:, :, 2] = new_vert[:, :, 2] * scale[2] * (1 - eps)

        mSphere = mSphere.update_padded(new_vert)
        return mSphere

# class adversarial_patch_3d(ABC):
    
#     def __init__(self, scale:list=[0.7, 0.7, 0.5]):
#         self.scale = np.array(scale)
        
#         self.offset_limit: torch.Tensor = torch.tensor([0.1]).cuda()
#         self.global_translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]).cuda()
#         self.global_translation.requires_grad_(True)
        
#         self.base_coord: torch.Tensor = torch.tensor([0.0, 0.0, -scale[2]]).float().cuda()
        
#         self.theta: torch.Tensor = torch.tensor([0.0]).cuda()
#         self.theta.requires_grad_(True)


class adversarial_patch_3d(ABC):
    
    def __init__(self, scale:list=[0.7, 0.7, 0.5],
                    eps = 0.0):
        self.scale = np.array(scale, dtype=np.float32)
        
        self.offset_limit: torch.Tensor = torch.tensor([0.1]).cuda()
        self.global_translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]).cuda()
        self.global_translation.requires_grad_(True)
        
        self.base_coord: torch.Tensor = torch.tensor([0.0, 0.0, -self.scale[2]]).float().cuda()
        
        self.theta: torch.Tensor = torch.tensor([0.0]).cuda()
        self.theta.requires_grad_(True)
        
    def get_parameters(self) -> list[torch.Tensor]:
        parameters = [self.global_translation, self.theta]
        return parameters
    
    def load_parameter(self, parameters:list) -> None:
        self.global_translation = parameters[0].clone().detach().to(self.global_translation.device)
        self.theta = parameters[1].clone().detach().to(self.theta.device)
        self.global_translation.requires_grad_(True)
        self.theta.requires_grad_(True)
        
    @abstractmethod
    def get_basic_meshes(self) -> Meshes:
        return None
        
    @abstractmethod
    def get_deformed_meshes(self) -> Meshes:
        return None
    
    @abstractmethod
    def get_transformed_meshes(self, 
                                pos:torch.Tensor,
                                theta:torch.Tensor,
                                adversarial_parameters:list[torch.Tensor]) -> Meshes:
        return None
    
    @abstractmethod
    def get_regularization_loss(self) -> torch.Tensor:
        return None
    
    def constrain_grad(self):
        if self.global_translation.grad is not None:
            self.global_translation.grad[2] = 0.
        

class single_sphere(adversarial_patch_3d):
    
    def __init__(self, scale:list=[0.7, 0.7, 0.5],
                    level:int = 2,
                    eps = 0.0):
        super().__init__(scale=scale,
                         eps=eps)
        
        self.sphere = learnable_sphere(scale = self.scale, 
                                       level = level,
                                       eps = eps)
            
    def get_regularization_loss(self):
        deformed_mesh = self.get_deformed_meshes()
        return mesh_laplacian_smoothing(deformed_mesh)
            
    def get_parameters(self) -> list[torch.Tensor]:
        parameters = super().get_parameters()
        parameters.append(self.sphere.deform_vert_logit)
        return parameters
    
    def load_parameter(self, parameters:list):
        super().load_parameter(parameters)
        self.sphere.deform_vert_logit = parameters[2].clone().detach().to(self.sphere.deform_vert_logit.device)
        self.sphere.deform_vert_logit.requires_grad_(True)
        
    def get_basic_meshes(self) -> Meshes:
        return self.sphere.get_basic_meshes()

    def get_deformed_meshes(self,
                            deform_vert_logit:torch.Tensor = None,) -> Meshes:
        return self.sphere.get_deformed_meshes(deform_vert_logit = deform_vert_logit)
    
    def get_transformed_meshes(self, 
                                pos:torch.Tensor,
                                theta:torch.Tensor,
                                adversarial_parameters:torch.Tensor = None,) -> Meshes:
        """
        Args:
            pos: [3,]
            theta: [1,]
        """

        deform_vert_logit = None
        global_translation = self.global_translation
        global_theta = self.theta
        if adversarial_parameters is not None:

            global_translation = adversarial_parameters[0]
            global_theta = adversarial_parameters[1]
            deform_vert_logit = adversarial_parameters[2]
            
            
        deformed_mesh = self.get_deformed_meshes(deform_vert_logit = deform_vert_logit)
        verts = deformed_mesh.verts_padded()
        verts = verts + (self.offset_limit * 
                         torch.tanh(global_translation / self.offset_limit))[None, :]
        global_R = self.generate_rotate_matrix(global_theta)
        verts = torch.matmul(verts, global_R.T)
        
        # translate to the rooftop of the vehicle
        local_R = self.generate_rotate_matrix(theta)
        verts = torch.matmul(verts, local_R.T)
        verts = verts + (pos - self.base_coord)
        
        transformed_mesh = deformed_mesh.update_padded(verts)
        
        return transformed_mesh
    
    def constrain_grad(self):
        super().constrain_grad()
        if self.sphere.deform_vert_logit.grad is not None:
            self.sphere.deform_vert_logit.grad[:, 2] = 0.
        
    def generate_rotate_matrix(self, theta:torch.Tensor) -> torch.Tensor:
        tensor_0 = torch.zeros(1).cuda()
        RZ = euler_angles_to_matrix(torch.concatenate([tensor_0, tensor_0, theta]), ["X", "Y", "Z"])

        return RZ

class simple_cubic_lattice:
    
    def __init__(self, cubic_level:int = 2,
                    scale:list=[0.7, 0.7, 0.5],
                    eps = 0.0):
        self.internal_atom: list[learnable_sphere] = None
        self.scale = np.array(scale)
        self.cubic_level = cubic_level
        
        self.offset_limit: torch.Tensor = torch.tensor([0.1]).cuda()
        self.global_translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]).cuda()
        self.global_translation.requires_grad_(True)
        
        self.base_coord: torch.Tensor = torch.tensor([0.0, 0.0, -scale[2]]).float().cuda()
        
        self.theta: torch.Tensor = torch.tensor([0.0]).cuda()
        self.theta.requires_grad_(True)
        
        self._init_internal_atoms()
        self.lattice_grid = torch.stack([grid_pos.contiguous().view(-1) for grid_pos in self.generate_mesh_grid()], dim=1)
        
    def _init_internal_atoms(self):
        dscale = self.scale / self.cubic_level
        self.internal_atoms = []
        for _ in range(self.cubic_level ** 3):
            self.internal_atoms.append(learnable_sphere(level = 2,
                                                        scale = dscale))
            
    def get_laplacian_loss(self):
        deformed_mesh = self.get_deformed_lattice()
        return mesh_laplacian_smoothing(deformed_mesh)
            
    def get_parameters(self):
        parameters = [self.global_translation, self.theta]
        for internal_atom in self.internal_atoms:
            parameters.append(internal_atom.deform_vert_logit)
        
        return parameters
    
    def load_parameter(self, parameters:list):
        self.global_translation = parameters[0]
        self.theta = parameters[1]
        for i, internal_atom in enumerate(self.internal_atoms):
            internal_atom.deform_vert_logit = parameters[2 + i]
            
    def generate_mesh_grid(self):
        dscale = self.scale/self.cubic_level
        x = torch.arange(-self.scale[0] + dscale[0], 
                         self.scale[0], dscale[0] * 2.0).cuda()
        y = torch.arange(-self.scale[1] + dscale[1], 
                         self.scale[1], dscale[1] * 2.0).cuda()
        z = torch.arange(-self.scale[2] + dscale[2], 
                         self.scale[2], dscale[2] * 2.0).cuda()
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
        return grid_x, grid_y, grid_z

    def get_deformed_lattice(self) -> Meshes:
        atoms_meshes = []
        for i in range(self.lattice_grid.size(0)):
            atoms_meshes.append(self.internal_atoms[i].get_transformed_meshes(self.lattice_grid[i]))
            # atoms_meshes.append(self.internal_atoms[i].get_deformed_meshes())
        atoms_meshes = join_meshes_as_batch(atoms_meshes)
        return atoms_meshes
    
    def get_transformed_meshes(self, 
                                pos:torch.Tensor,
                                theta:torch.Tensor) -> Meshes:
        deformed_mesh = self.get_deformed_lattice()
        verts = deformed_mesh.verts_padded()
        verts = verts + (self.offset_limit * 
                         torch.tanh(self.global_translation / self.offset_limit))[None, :]
        global_R = self.generate_rotate_matrix(self.theta)
        verts = torch.matmul(verts, global_R.T)
        
        # translate to the rooftop of the vehicle
        local_R = self.generate_rotate_matrix(theta)
        verts = torch.matmul(verts, local_R.T)
        verts = verts + (pos - self.base_coord)
        
        transformed_mesh = deformed_mesh.update_padded(verts)
        
        return transformed_mesh
    
    def constrain_z_grad(self):
        self.global_translation.grad[2] = 0.
        for internal_atom in self.internal_atoms:
            internal_atom.deform_vert_logit.grad[:, 2] = 0.
        
    def generate_rotate_matrix(self, theta:torch.Tensor) -> torch.Tensor:
        tensor_0 = torch.zeros(1).cuda()
        RZ = euler_angles_to_matrix(torch.concatenate([tensor_0, tensor_0, theta]), ["X", "Y", "Z"])

        return RZ