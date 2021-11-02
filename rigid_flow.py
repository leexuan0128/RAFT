import sys
sys.path.append("reprojection")

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

torch.set_default_dtype(torch.float32)

def get_rigid_flow(disparity, count):
    # get depth
    fx, K, inv_K = functional_data.read_focal_length(15)
    #B = functional_data.read_baseline('Transformation.txt')
    #T = functional_data.read_transformation('Transformation.txt').astype('float32')
    T, B = functional_data.read_T_and_B('camera_data.txt', count)
    T = T[np.newaxis, :, :]
    T = torch.from_numpy(T)
    height, width = disparity.shape
    depths = depth.create_depths_map(disparity, fx, B, height, width)

    #get rigid flow (x2 - x1)
    reprojections = Reprojection(height, width)
    x2 = reprojections.forward(depths, T, K, inv_K)
    x1 = functional_data.read_original_coords(544, 960)
    rigid = x2 - x1
    rigid = torch.squeeze(rigid) # new x2 2D coordinate
    rigid = rigid.numpy().astype('float32')
    filename = "raft_pred_dis_rigid_flow_{}.flo".format(count)
    IO.writeFlow(filename, rigid)
    flow2img.visulize_flow_file(filename)

    return rigid
    
if __name__ == '__main__':
    get_rigid_flow()
