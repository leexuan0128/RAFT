import sys
sys.path.append(r"./reprojections")

from backprojection import Backprojection
from transformation3d import Transformation3D
from projection import Projection
from reprojection import Reprojection

import flow2img
import IO
import depth
import functional_data
import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

torch.set_default_dtype(torch.float64)

def rigid_flow():
    # get depth
    fx, K, inv_K = read_focal_length(15)
    T = functional_data.read_transformation('Transformation.txt')
    T = T[np.newaxis, :, :]
    T = torch.from_numpy(T)
    B = functional_data.read_baseline('Transformation.txt')
    disparity_path_dir = Path(r"datasets/Driving/disparity")
    pfm_images = disparity_path_dir.joinpath('0401.pfm')
    depth = depth.create_depths_map(pfm_images)

    #get rigid flow, that is x2 - x1
    reprojections = Reprojection(540,960)
    x2 = reprojections.forward(depth, T, K, inv_K)
    x1 = functional_data.read_original_coords(540, 960)
    rigid_flow =  x2 - x1
    rigid_flow = torch.squeeze(rigid_flow)
    rigid_flow = rigid_flow.numpy().astype('float32')

    filename = "rigid_flow_back.flo"
    IO.writeFlow(filename, rigid_flow)
    #flow2img.visulize_flow_file(filename)
    
if __name__ == '__main__':
    rigid_flow()
