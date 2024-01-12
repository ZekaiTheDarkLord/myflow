import torch
import torch.nn.functional as F

import utils
import time


def global_correlation_softmax(feature0, feature1,
                               bw_flow=False,
                               ):
    # global correlation
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, H*W]

    init_grid = utils.coords_grid(b, h, w).to('cuda')  # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    # torch.cuda.synchronize()
    # start_time = time.time()
    correlation = feature0 @ feature1 / (c ** 0.5)  # [B, H*W, H*W]
    # torch.cuda.synchronize()
    # print(correlation.shape)
    # end_time = time.time()
    # print('global_correlation_softmax:', end_time-start_time)

    if bw_flow:
        correlation = torch.cat((correlation, correlation.permute(0, 2, 1)), dim=0)  # [2*B, H*W, H*W]
        init_grid = init_grid.repeat(2, 1, 1, 1)  # [2*B, 2, H, W]
        grid = grid.repeat(2, 1, 1)  # [2*B, H*W, 2]
        b = b * 2
    # else:
    #     curr_init_grid = init_grid
    #     curr_grid = grid

    prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W]

    # print(prob.shape)
    # print(grid.shape)
    # print('==')

    correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
    flow = correspondence - init_grid

    # if w == 80:
    #     global test_index
    #     test_attn = torch.max(correlation.abs(), dim=-1)
    #     test_attn = test_attn.view(b, h, w)
    #     test_attn = test_attn[0].cpu().numpy()

    #     test_attn = np.uint8((test_attn-test_attn.min())/(test_attn.max()-test_attn.min())*255)
    #     test_attn = cv2.resize(test_attn, (640, 512))
    #     print(test_attn)
        
    #     cv2.imwrite('/data/gmflow/output/'+str(test_index)+'_attn.png', test_attn)
    #     test_index += 1

    return flow

def local_correlation_softmax(feature0, feature1, local_radius):
    b, c, h, w = feature0.size()

    torch.cuda.synchronize()
    start_time = time.time()

    coords_init = utils.coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = utils.generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

    sample_coords_softmax = sample_coords

    # exclude coords that are out of image space
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [B, H*W, (2R+1)^2]

    valid = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

    # normalize coordinates to [-1, 1]
    sample_coords_norm = utils.normalize_coords(sample_coords, h, w)  # [-1, 1]

    torch.cuda.synchronize()
    end_time = time.time()
    print('sample_coords_norm:', end_time-start_time)
        
    window_feature = F.grid_sample(feature1, sample_coords_norm,
                                   padding_mode='zeros', align_corners=True
                                   ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]

    # mask invalid locations
    corr[~valid] = -1e9

    prob = F.softmax(corr, -1)  # [B, H*W, (2R+1)^2]

    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(
        b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    flow = correspondence - coords_init

    return flow