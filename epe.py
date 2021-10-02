# calculate the avg end point error: target value - GT value

import torch
import numpy as np
import IO
import matplotlib.pyplot as plt
from utils.utils import InputPadder

def get_epe(all_flow, gt_filename):
    # GT imgsave
    gt_flow = IO.readFlow(gt_filename)
    #flo = flow_viz.flow_to_image(gt_flow)
    #plt.imsave("GT_401_back.png", flo)
    gt_flow = torch.Tensor([gt_flow])
    gt_flow = gt_flow.permute(0,3,1,2)
    padder = InputPadder(gt_flow.shape)
    gt_flow = padder.pad(gt_flow)[0]
    gt_flow = gt_flow.numpy()

    all_epe = []
    for flow in all_flow:
        flow = flow.numpy()
        epe = np.mean(np.sum((flow - gt_flow)**2, axis = 1)**0.5) # Euclidean distance 
        #epe = np.sum((flow - gt_flow)**2, axis = 1)**0.5
        all_epe.append(epe)
    #np.save("non_rigid_avg_epe_back_15.npy", all_epe)
    # flowList = np.load("non_rigid_avg_epe_back_15.npy")
    #plt.imshow(epe[0], vmin=0, vmax=5)
    #plt.savefig("err.png")

    # draw
    x = range(len(all_epe))
    y = all_epe
    #y2 = flowList
    plt.title("{} Iterations EPE".format(str(len(all_epe)))) 
    plt.xlabel("Iterations") 
    plt.ylabel("avg end-point-error")
    plt.grid(True)
    plt.plot(x,y, label = "rigid")
    #plt.plot(x,y2, label = "non-rigid")
    plt.legend()
    plt.savefig("end_point_error/rigid_avg_epe_back_{}.png".format(str(len(all_epe))))
    plt.show()