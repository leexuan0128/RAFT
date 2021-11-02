# calculate the avg end point error: target value - GT value

import torch
import numpy as np
import IO
import matplotlib.pyplot as plt
from utils.utils import InputPadder

def get_epe(flow, gt_filename):
    gt_flow = IO.readFlow(gt_filename)
    #flo = flow_viz.flow_to_image(gt_flow)
    #plt.imsave("GT_401_back.png", flo)
    gt_flow = torch.Tensor([gt_flow])
    gt_flow = gt_flow.permute(0,3,1,2)
    padder = InputPadder(gt_flow.shape)
    gt_flow = padder.pad(gt_flow)[0]
    # gt_flow = gt_flow.numpy()

    # all_epe = []
    # for flow in all_flow:
    #     flow = flow.cpu().numpy()
    #     epe = np.mean(np.sum((flow - gt_flow)**2, axis = 1)**0.5) # Euclidean distance 
    #     # epe = np.sum((flow - gt_flow)**2, axis = 1)**0.5
    #     all_epe.append(epe)

    epe = torch.sum((flow.cpu() - gt_flow)**2, axis = 1).sqrt()
    
    # np.save("tmp/raft_avg_epe_gtdis_15.npy", all_epe)
    # flowList = np.load("tmp/raft_avg_epe_gtdis_15.npy")
    # plt.imshow(epe[0], vmin=0, vmax=5)
    # plt.savefig("err_b_10.png")

    # draw plots
    # x = list(range(len(all_epe)))
    # y = all_epe
    # plt.clf()
    # plt.cla()
    # y2 = list(flowList)
    # plt.title("{} Iterations EPE".format(str(len(all_epe)))) 
    # plt.xlabel("Iterations") 
    # plt.ylabel("average end-point-error")
    # plt.grid(True)
    # plt.plot(x,y, label="raft_rigid_flow_pred_disparity")
    # plt.plot(x,y2, label = "raft_rigid_flow_gt_disparity")
    # plt.legend()
    # # plt.savefig("final_epe/raft_{}.png".format(i))
    # # plt.savefig("final_epe/mix_{}.png".format(i))
    # plt.savefig("final_epe/raft_gt_dis_{}.png".format(i))
    # # plt.savefig("final_epe/raft_pred_dis_{}.png".format(i))
    
    return epe