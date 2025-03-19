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
