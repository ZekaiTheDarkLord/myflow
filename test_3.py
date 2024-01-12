import torch
import os
import numpy as np
import cv2
import glob

from myflow import GMFlow
from data_utils.frame_utils import readPFM
from utils import forward_backward_consistency_check

import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# torch.set_printoptions(precision=8)
# torch.backends.cudnn.benchmark = True

myflow_model = GMFlow().to('cuda')
myflow_model.init_hw(48, 64)

checkpoint = torch.load('/data/gmflow/pretrained/temp_ctf3_v2.pth', map_location='cuda')

state_dict = {}
for k, v in checkpoint['model'].items():

    state_dict[k] = v

# init_weights = torch.tensor([[0, 0, 0],
#                              [0, 1, 0],
#                              [0, 0, 0]]).float().view(1, 1, 3, 3).repeat(128, 128, 1, 1)
# state_dict['backbone.conv.weight'] = init_weights

myflow_model.load_state_dict(state_dict, strict=True)
# print(myflow_model)

myflow_model.eval()

img_0 = np.ones((384,512,3),np.uint8)*np.array([0,255,0])
img_1 = np.ones((384,512,3),np.uint8)*np.array([0,255,0])

img_0[100:200,100:300] = (0,0,255)
img_1[100:200,200:400] = (0,0,255)

cv2.imwrite('/data/gmflow/temp_output/img_0.jpg', img_0)
cv2.imwrite('/data/gmflow/temp_output/img_1.jpg', img_1)

@torch.no_grad()
def inference(img_path_0, img_path_1, flow_0_path=None, flow_1_path=None):

    img0 = cv2.imread(img_path_0)
    img1 = cv2.imread(img_path_1)

    # img0 = cv2.resize(img0, (640, 512))
    # img1 = cv2.resize(img1, (640, 512))
    img0 = cv2.resize(img0, (512, 384))
    img1 = cv2.resize(img1, (512, 384))

    img0_cuda = torch.from_numpy(img0).permute(2, 0, 1).float()[None].cuda()
    img1_cuda = torch.from_numpy(img1).permute(2, 0, 1).float()[None].cuda()

    torch.cuda.synchronize()
    
    start_time = time.time()

    flow_list = myflow_model(img0_cuda, img1_cuda)
    torch.cuda.synchronize()
    print(flow_list[-2].shape)
    print(flow_list[-1].shape)
    end_time = time.time()
    print('Total:', end_time-start_time)
    print('===')

    flow = flow_list[-1][0]

    flow = flow.permute(1,2,0).cpu().numpy()
    print(flow)
    flow = np.sqrt(np.sum(flow**2, axis=-1))

    flow = np.uint8((flow-flow.min())/(flow.max()-flow.min())*255)

    flow = cv2.resize(flow, (512, 384))
    cv2.imwrite('/data/gmflow/temp_output/flow.jpg', flow)


inference('/data/gmflow/temp_output/img_0.jpg', '/data/gmflow/temp_output/img_1.jpg')
# inference('/data/droid_slam/temp_lcd/9.jpg', '/data/droid_slam/temp_lcd/10.jpg')
