from pathlib import Path
import numpy as np
import csv
import re
import cv2
import random
import math
import matplotlib.pyplot as plt

from functional_data import *

# def read_pfm(file):
#     file = open(file, 'rb')
#     header = file.readline().decode().rstrip()
#     channels = 3 if header == 'PF' else 1
#     dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('ascii'))
#     if dim_match:
#         width, height = list(map(int, dim_match.groups()))
#     else:
#         raise Exception("Malformed PFM header.")
#     scale = float(file.readline().decode("ascii").rstrip())
#     if scale < 0:
#         endian = '<'  # littel endian
#         scale = -scale
#     else:
#         endian = '>'  # big endian
    
#     data = np.fromfile(file, endian + 'f') 
#     data = np.reshape(data, newshape=(height, width))
#     data = np.flipud(data)
#     #np.savetxt(r"./dispariy.txt", data, fmt='%f', delimiter=' ')
#     disparity_img = np.reshape(data, newshape=(height, width, channels))
#     # disparity_img = np.flipud(disparity_img).astype('float32')
#     cv2.imwrite("tmp/disparity_map.png", disparity_img)

#     return data, [(height, width, channels), scale]

def create_depths_map(disparity, focal_length, baseline, height, width):
    
    depth_map = (focal_length * baseline) / abs(disparity)
    # depth_img = np.reshape(depth_map, newshape=(height, width, 1))
    # depth_img = np.flipud(depth_map).astype('float32')
    # depth_img = np.flipud(depth_img).astype('uint8')     
    # plt.imshow(depth_img, cmap='gray')
    # plt.savefig("tmp/depth_map_raft.png")
    # plt.imsave("tmp/depth_map_raft.png", depth_map, cmap="gray")
    # cv2.imwrite("tmp/depth_map_raft.png", depth_map)
    depth_map = depth_map[np.newaxis, np.newaxis, :, :]

    # depth_map (Nx1xHxW)
    return depth_map

