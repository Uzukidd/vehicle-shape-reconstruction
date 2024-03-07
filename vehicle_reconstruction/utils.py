import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model_bbox(vertices: np.ndarray):
    xmin, xmax = vertices[:, 0].min(), vertices[:, 0].max()
    ymin, ymax = vertices[:, 1].min(), vertices[:, 1].max()
    zmin, zmax = vertices[:, 2].min(), vertices[:, 2].max()
    length = xmax - xmin
    width = ymax - ymin
    height = zmax - zmin

    return (length, width, height, xmin, xmax, ymin, ymax, zmin, zmax)

