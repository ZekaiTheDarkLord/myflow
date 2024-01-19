import torch
import torch.nn.functional as F

import backbone
import transformer
import matching
import refine
import utils

import config
import time


class MyFlow(torch.nn.Module):
    def __init__(self):
        super(MyFlow, self).__init__()
        self.backbone = backbone.CNNEncoder()
        self.cross_attn_s2 = transformer.FeatureAttention(config.feature_dim+2, num_layers=2, bidir=True, ffn=True, ffn_dim_expansion=1, post_norm=True)
        
        self.matching_s2 = matching.Matching()

        self.flow_attn_s2 = transformer.FlowAttention(config.feature_dim+2)

        # self.feature_interp_conv = torch.nn.ConvTranspose2d(config.feature_dim+2, config.feature_dim+2, kernel_size=4, stride=2, padding=1, bias=True)

        self.merge_conv = torch.nn.Sequential(torch.nn.Conv2d((config.feature_dim+2) * 2, config.feature_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.GELU(),
                                              torch.nn.Conv2d(config.feature_dim * 2, config.feature_dim, kernel_size=3, stride=1, padding=1, bias=False))

        # self.flow_interp_conv = torch.nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=True)

        self.refine_s1 = refine.Refine(config.feature_dim, patch_size=7, num_layers=6)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def init_bhw(self, batch_size, height, width, device):
        self.backbone.init_pos_12(batch_size, height//8, width//8, device)
        self.matching_s2.init_grid(batch_size, height//16, width//16, device)
        utils.init_mean_std(device)

    def forward(self, img0, img1):

        flow_list = []

        img0 = utils.normalize_img(img0)
        img1 = utils.normalize_img(img1)

        feature0_list = self.backbone(img0)
        # torch.cuda.synchronize()
        # start_time = time.time()
        feature1_list = self.backbone(img1)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('backbone:', end_time-start_time)

        feature0_s1, feature0_s2 = feature0_list
        feature1_s1, feature1_s2 = feature1_list

        feature0_s2, feature1_s2 = self.cross_attn_s2(feature0_s2, feature1_s2)
        flow0 = self.matching_s2.global_correlation_softmax(feature0_s2, feature1_s2)

        flow0 = self.flow_attn_s2(feature0_s2, flow0)
        flow_list.append(flow0)

        feature0_s2 = F.interpolate(feature0_s2, scale_factor=2, mode='nearest')
        feature1_s2 = F.interpolate(feature1_s2, scale_factor=2, mode='nearest')

        feature0_s1 = self.merge_conv(torch.cat([feature0_s1, feature0_s2], dim=1))
        feature1_s1 = self.merge_conv(torch.cat([feature1_s1, feature1_s2], dim=1))

        flow0 = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2

        feature1_s1 = utils.flow_warp(feature1_s1, flow0)

        delta_flow = self.refine_s1(feature0_s1, feature1_s1, flow0)
        flow0 = flow0 + delta_flow

        flow_list.append(flow0)

        return flow_list
