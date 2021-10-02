# get flow_init (rigid_flow)

import IO
import cv2
import numpy as np
import torch
from utils.utils import InputPadder

def get_flow_init(filename):
    flow_init = IO.readFlow(filename)
    flow_init = torch.Tensor([flow_init]).permute(0,3,1,2)
    
    padder = InputPadder(flow_init.shape)
    flow_init = padder.pad(flow_init)[0]
    width = int(flow_init.shape[3] / 8)
    height = int(flow_init.shape[2] / 8)

    flow_init = flow_init.permute(0,2,3,1)
    flow_init = torch.squeeze(flow_init, 0)
    flow_init = flow_init.numpy()
    flow_init = cv2.resize(flow_init, (width, height), interpolation = cv2.INTER_CUBIC) / 8
    flow_init = torch.Tensor([flow_init]).permute(0,3,1,2)

    return flow_init