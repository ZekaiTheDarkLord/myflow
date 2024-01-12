import torch
import torch.nn.functional as F

import utils
import time


class Matching:

    def __init__(self, bw_flow=False):
        self.bw_flow = bw_flow

    def init_grid(self, batch_size, height, width):
        self.grid = utils.coords_grid(batch_size, height, width).to('cuda')  # [B, 2, H, W]
        self.flatten_grid = self.grid.view(batch_size, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

        # if bw_flow:
        #     correlation = torch.cat((correlation, correlation.permute(0, 2, 1)), dim=0)  # [2*B, H*W, H*W]
        #     self.grid = self.grid.repeat(2, 1, 1, 1)  # [2*B, 2, H, W]
        #     flatten_grid = flatten_grid.repeat(2, 1, 1)  # [2*B, H*W, 2]
        #     b = b * 2

    def global_correlation_softmax(self, feature0, feature1):

        b, c, h, w = feature0.shape

        feature0 = feature0.flatten(-2).permute(0, 2, 1)
        feature1 = feature1.flatten(-2).permute(0, 2, 1)

        correspondence = F.scaled_dot_product_attention(feature0, feature1, self.flatten_grid)

        correspondence = correspondence.view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        flow = correspondence - self.grid

        return flow

    def local_correlation_softmax(self, feature0, feature1, local_radius):
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