import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import IO
import epe
import init_flow

from PIL import Image
from raft import *
from utils import flow_viz
from utils.utils import InputPadder

# DEVICE = 'cuda'
DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(img, flo):
    #origin img
    img = img[0].permute(1,2,0).cpu().numpy()
    #flow img
    flo = flo[0].permute(1,2,0).cpu().numpy()
    #IO.writeFlow("flowdata/optical_flow.flo", flo)
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # img_flo = np.concatenate([img, flo], axis=0)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    #cv2.waitKey()
    #cv2.imwrite("tmp/img_flow_{}.jpg".format(i), img_flo)

    return flo

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location = torch.device('cpu'))) # cpu model
    # model.load_state_dict(torch.load(args.model)) # gpu model
    model = model.module
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        # call flow_init
        flow_init = init_flow.get_flow_init("rigid_flow_back.flo")

        images = sorted(images)
        i = 1
        for imfile1, imfile2 in zip(images[1:], images[:-1]): # backward flow
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            #padding img to be divided by 8
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            #flow_low: 1/8 flow, flow_up: full resolution flow, flow_all: all iters flow
            flow_low, flow_up, flow_all = model(image1, image2, flow_init, iters=20, test_mode=True)
            flow_image = viz(image1, flow_up)
            # cv2.imwrite("tmp/raft_optical_flow_{}.jpg".format(i), flow_image)
            plt.imsave("tmp/rigid_optical_flow_back_new1_{}.jpg".format(i), flow_image)
            i = i + 1
    #evaluation via epe
    epe.get_epe(flow_all, "GT-B-0401.pfm")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)