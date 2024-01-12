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

# checkpoint = torch.load('/data/gmflow/pretrained/\.pth', map_location='cuda')

# state_dict = {}
# for k, v in checkpoint['model'].items():

#     state_dict[k] = v

# init_weights = torch.tensor([[0, 0, 0],
#                              [0, 1, 0],
#                              [0, 0, 0]]).float().view(1, 1, 3, 3).repeat(128, 128, 1, 1)
# state_dict['backbone.conv.weight'] = init_weights

# myflow_model.load_state_dict(state_dict, strict=True)
# print(myflow_model)

myflow_model.eval()

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
    print(flow_list[-1].shape)
    end_time = time.time()
    print('Total:', end_time-start_time)
    print('===')

    # fwd_flow = flows[0].unsqueeze(0)

    # fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow)

    # flow = fwd_flow[0].permute(1,2,0).cpu().numpy()
    # flow = np.sqrt(np.sum(flow**2, axis=-1))

    # flow = np.uint8((flow-flow.min())/(flow.max()-flow.min())*255)

    # flow = cv2.resize(flow, (640, 480))
    # file_name = os.path.splitext(os.path.basename(img_path_0))[0]
    # cv2.imwrite('/data/gmflow/temp_output/'+file_name+'_my.jpg', cv2.applyColorMap(flow, cv2.COLORMAP_VIRIDIS))
    # cv2.imwrite('/data/gmflow/temp_output/'+file_name+'_img.jpg', img0)

    # plt.subplot(131)
    # plt.imshow(img0)
    # plt.subplot(132)
    # plt.imshow(flow)
    # plt.subplot(133)
    # plt.imshow(img1)
    # plt.show()

    # torch_flow_gt = torch.from_numpy(readPFM(flow_0_path)[:, :, :-1].astype(np.float32))
    # # flow_1_gt = torch.from_numpy(readPFM(flow_1_path)[:, :, :-1].astype(np.float32))

    # ht, wd = torch_flow_gt.shape[:2]

    # ys, xs = torch.meshgrid(torch.arange(ht),
    #                         torch.arange(wd), indexing='ij')

    # torch_flow_gt[:,:,0] += xs
    # torch_flow_gt[:,:,1] += ys

    # torch_flow_gt[:, :, 0] = 2 * torch_flow_gt[:, :, 0] / (wd - 1) - 1
    # torch_flow_gt[:, :, 1] = 2 * torch_flow_gt[:, :, 1] / (ht - 1) - 1

    # torch_img = torch.from_numpy(img1).permute(2, 0, 1).float()

    # torch_warp_img = F.grid_sample(torch_img[None], torch_flow_gt[None])
    
    # torch_warp_img = torch_warp_img[0].permute(1, 2, 0).numpy().astype(np.uint8)

    # flow_0_gt = readPFM(flow_0_path)[:, :, :-1].astype(np.float32)
    # flow_1_gt = readPFM(flow_1_path)[:, :, :-1].astype(np.float32)

    # x_range = np.arange(flow_0_gt.shape[1])
    # y_range = np.arange(flow_0_gt.shape[0])
    # xs, ys = np.meshgrid(x_range, y_range)
    # coords = np.float32(np.dstack([xs, ys]))

    # warp_flow = cv2.remap(coords, coords+flow_1_gt, None, interpolation=cv2.INTER_LINEAR)
    # warp_flow = cv2.remap(warp_flow, coords+flow_0_gt, None, interpolation=cv2.INTER_LINEAR)

    # warp_flow -= coords

    # warp_flow = np.sum(warp_flow**2, axis=-1) < 0.01

    # flow_0_gt[~warp_flow] = -1000

    # warp_img = cv2.remap(img1, coords+flow_0_gt, None, interpolation=cv2.INTER_LINEAR)

    # plt.subplot(131)
    # plt.imshow(img0)
    # plt.subplot(132)
    # plt.imshow(warp_img)
    # plt.subplot(133)
    # plt.imshow(img1)
    # plt.show()



# inference('datasets/FlyingThings3D/frames_finalpass/TRAIN/B/0654/left/0007.png', 
#         'datasets/FlyingThings3D/frames_finalpass/TRAIN/B/0654/left/0008.png', 
#         'datasets/FlyingThings3D/optical_flow/TRAIN/B/0654/into_future/left/OpticalFlowIntoFuture_0007_L.pfm',
#         'datasets/FlyingThings3D/optical_flow/TRAIN/B/0654/into_past/left/OpticalFlowIntoPast_0008_L.pfm')
inference('/data/droid_slam/temp_lcd/6.jpg', '/data/droid_slam/temp_lcd/7.jpg')

start_time = time.time()

img_path_list = sorted(glob.glob('/data/droid_slam/zed/left_cam/*.jpg'))[100::10][:200]

for src_img_path, dst_img_path in zip(img_path_list[:-1], img_path_list[1:]):
    inference(src_img_path, dst_img_path)
    # break

end_time = time.time()
print(end_time-start_time)