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
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    # print(flo)
    IO.writeFlow("raft_optical_flow.flo", flo)
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    #cv2.waitKey()
    #cv2.imwrite("tmp/img_flow_{}.jpg".format(i), img_flo)
    # return img_flo
    return flo

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location = torch.device('cpu')))
    # model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        flow_init = IO.readFlow("/home/xuanli/RAFT/rigid_flow.flo")
        flow_init = cv2.resize(flow_init,(120,68)) # interpolation = cv2.INTER_CUBIC
        flow_init = flow_init[np.newaxis, :, :, :]
        flow_init = torch.from_numpy(flow_init).permute(0,3,1,2)

        images = sorted(images)
        i = 1
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image1, image2, flow_init, iters=14, test_mode=True)
            # viz(image1, flow_up)
            flow_image = viz(image1, flow_up)
            # cv2.imwrite("tmp/raft_optical_flow_{}.jpg".format(i), flow_image)
            plt.imsave("tmp/raft_optical_flow_{}.jpg".format(i), flow_image)
            # cv2.imshow("image", flow_image[:, :, [2,1,0]]/255.0)
            # cv2.waitkey()
            i = i + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)

# if __name == '__main__':
#     demo()
