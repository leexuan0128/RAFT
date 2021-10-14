from pathlib import Path
import numpy as np
import csv
import re
import cv2
import random
import math
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

def read_transformation(transformation_path):
    with open(transformation_path, 'r') as f:
        lines = f.readlines()
        T1_data = lines[0].split(' ')
        T2_data = lines[2].split(' ')
        T1_data = np.array(T1_data, dtype=float).reshape(4,4)
        T2_data = np.array(T2_data, dtype=float).reshape(4,4)

        inv_T1_data = np.linalg.inv(T1_data)
        # inv_T2_data = np.linalg.inv(T2_data) 
        T_21 = np.dot(inv_T1_data, T2_data)
        
    return T_21

def read_focal_length(focal_length):
    if focal_length == 15:
        K = [[450.0, 0.0, 479.5, 0.0],
            [0.0, 450.0, 269.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
        fx = K[0][0]
        K = np.array(K).astype('float32')
        K = K[np.newaxis, :, :]
        K = torch.from_numpy(K)
        inv_K = np.linalg.inv(K).astype('float32')
        inv_K = torch.from_numpy(inv_K)

    if focal_length == 35:
        K = [[1050.0, 0.0, 479.5, 0.0],
            [0.0, 1050.0, 269.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
        fx = K[0][0]
        K = np.array(K)
        K = K[np.newaxis, :, :]
        K = torch.from_numpy(K)
        inv_K = np.linalg.inv(K)
        inv_K = torch.from_numpy(inv_K)

    return fx, K, inv_K

def read_baseline(baseline_path):
    with open(baseline_path, 'r') as b:
        lines = b.readlines()
        T_L = lines[0].split(' ')
        T_R = lines[1].split(' ')
        T_L = np.array(T_L, dtype=float).reshape(4,4)
        T_R = np.array(T_R, dtype=float).reshape(4,4)
        inv_TR = np.linalg.inv(T_R)
        T_RL = np.dot(inv_TR, T_L)
        Baseline = abs(round(T_RL[0][3]))

    return Baseline

def read_original_coords(height, width):
    # generate regular grid
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.tensor(id_coords)

    # generate homogeneous pixel coordinates
    ones = nn.Parameter(torch.ones(1, 1, height * width),
                                requires_grad=False)
    xy = torch.unsqueeze(
                    torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0)
                    , 0)
    # xy = torch.cat([xy, ones], 1)
    xy = nn.Parameter(xy, requires_grad=False)

    # original xy
    xy = xy.view(1, 2, height, width)
    xy = xy.permute(0, 2, 3, 1)
    #print(xy)

    return xy

