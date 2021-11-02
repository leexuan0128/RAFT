import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rigid_flow
import init_flow
import read_gt_dis

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8, InputPadder

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)
        coords2 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1, coords2

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, stereo_image1, i, stereo_mode=False, flow_init=None, iters=12, upsample=True, test_mode=False):
        """ Estimate optical flow and depth between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        stereo_image1 = 2 * (stereo_image1 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        stereo_image1 = stereo_image1.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim
        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2, fmap3 = self.fnet([image1, image2, stereo_image1])      
            # fmap1, fmap3 = self.fnet([image1, stereo_image1])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        fmap3 = fmap3.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            #stereo_corr_fn = CorrBlock(fmap1, fmap3, radius=self.args.corr_radius, stereo_matching=True) # stereo matching corr
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius) # F corr 
            stereo_corr_fn = CorrBlock(fmap1, fmap3, radius=self.args.corr_radius, stereo_matching=True) # stereo matching corr

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1, coords2 = self.initialize_flow(image1)

        if stereo_mode:            
            # stereo matching for predicting depth
            for itr in range(iters):
                coords2 = coords2.detach()
                stereo_corr = stereo_corr_fn(coords2)
                disparity_flow = coords2 - coords0
                with autocast(enabled=self.args.mixed_precision):
                    net, up_mask, disparity_data = self.update_block(net, inp, stereo_corr, disparity_flow)
                # in stereo mode, project flow onto epipolar
                disparity_data[:,1] = 0.0
                coords2 = coords2 + disparity_data            
                # fold up by 8
                if up_mask is None:
                    disparity_up = upflow8(coords2 - coords0)
                else:
                    disparity_up = self.upsample_flow(coords2 - coords0, up_mask)    
            disparity = disparity_up[:,:1]
            # disparity = np.load("raft-stereo.npy")
            # disparity = read_gt_dis.read_pfm("gt_disparity/gt_dis_0401.pfm")

            disparity_img = -disparity
            disparity_img = disparity_img.cpu().numpy().squeeze()

            # disparity_img = cv2.resize(disparity_img, (960,544), interpolation=cv2.INTER_NEAREST)
            # disparity_img = torch.tensor(disparity_img.copy())
            # disparity_img = disparity_img.unsqueeze(0)
            # disparity_img = disparity_img.cpu().numpy().squeeze()
            # plt.imsave("final_disparity/disparity.{}.png".format(i), disparity_img, cmap="gray")
            

            # compute the rigid flow
            raft_rigid_flow = rigid_flow.get_rigid_flow(disparity_img, i)
            # Insert rigid flow as initialization
            raft_flow_init = init_flow.get_flow_init(raft_rigid_flow)
        
        # predict optical flow
        if flow_init is not None:
            coords1 = coords1 + raft_flow_init # add the initialization

        flow_predictions = []
        for itr in range(iters):   
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow # F(t+1) = F(t) + \Delta(t)
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up, flow_predictions, disparity_img
            
        return flow_predictions