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

torch.set_default_dtype(torch.float32)

DEVICE = 'cuda'
#DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)# map flow to rgb image

    return flo

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    #model.load_state_dict(torch.load(args.model, map_location = torch.device('cpu'))) # cpu model
    model.load_state_dict(torch.load(args.model)) # gpu model
    model = model.module
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        stereo_images = glob.glob(os.path.join(args.path2, '*.png')) + \
                        glob.glob(os.path.join(args.path2, '*.jpg'))
        gt_flow_all = glob.glob(os.path.join("gt_optical_flow", '*.pfm'))
        
        # call flow_init
        #flow_init = init_flow.get_flow_init("rigid_flow_back.flo")

        images = sorted(images)
        stereo_images = sorted(stereo_images)
        gt_flow_all = sorted(gt_flow_all)

        i = 0
        epe_list = []
        single_epe_list = []
        for imfile1, imfile2 in zip(images[1:], images[:-1]): #backward optical flow 
            # for stereoim1, stereoim2 in zip(stereo_images[1:], stereo_images[:-1]): # stereo pairs
            stereoim1 = stereo_images[i+1]
            stereo_image1 = load_image(stereoim1)
            #stereo_image2 = load_image(stereoim2)
            padder = InputPadder(stereo_image1.shape) # padding img to be divided by 8
            stereo_image1 = padder.pad(stereo_image1)[0]

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape) # padding img to be divided by 8
            image1, image2 = padder.pad(image1, image2)

            #flow_low: 1/8 flow, flow_up: full resolution flow, flow_all: all iters flow
            flow_low, flow_up, flow_all, disparity = model(image1, image2, stereo_image1, i, stereo_mode=True, flow_init=0, iters=15, test_mode=True)
            flow_image = viz(image1, flow_up)
            
            #plt.imsave("final_disparity/disparity_{}.png".format(i), disparity, cmap="jet")
            #depth_map = (450.0 * 1.0) / abs(disparity)
            #cv2.imwrite("final_depth/depth_{}.png".format(i), depth_map)
            #plt.imsave("final_optical_flow/raft_rigid_flow_back_{}.png".format(i), flow_image)
            
            epe_s = epe.get_epe(flow_up, gt_flow_all[i])
            epe_list.append(epe_s.view(-1).numpy())
            single_epe_list.append(np.mean(epe_s.view(-1).numpy()))

            i = i + 1

        epe_all = np.concatenate(epe_list)
        avg_epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        # print("Validation Avg EPE: %f" % avg_epe)
        print("Validation Avg EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (avg_epe, px1, px3, px5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--path2', help="dataset for stereo matching")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)