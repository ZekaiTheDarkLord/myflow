import torch
import torch.nn.functional as F

import backbone
import transformer
import matching
import up_sampler
import utils

import config
import time


class GMFlow(torch.nn.Module):
    def __init__(self):
        super(GMFlow, self).__init__()
        self.backbone = backbone.CNNEncoder()
        self.cross_attn_s2 = transformer.FeatureAttention(config.feature_dim+2, num_layers=3, bidir=True, ffn=True, ffn_dim_expansion=1, post_norm=True)
        
        self.matching_s2 = matching.Matching()

        self.flow_attn_s2 = transformer.FlowAttention(config.feature_dim+2)

        # torch.nn.ConvTranspose2d(config.feature_dim+2, config.feature_dim+2, kernel_size=2, stride=1, bias=True)

        self.merge_conv = torch.nn.Sequential(torch.nn.Conv2d((config.feature_dim+2) * 2, config.feature_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.GELU(),
                                              torch.nn.Conv2d(config.feature_dim * 2, config.feature_dim, kernel_size=3, stride=1, padding=1, bias=False))

        self.fine_up_sampler = up_sampler.Fine(config.feature_dim, patch_size=7)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def init_hw(self, batch_size, height, width):
        self.backbone.init_pos_12(batch_size, height, width)
        self.matching_s2.init_grid(batch_size, height//2, width//2)

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
        flow_list.append(flow0)

        flow0 = self.flow_attn_s2(feature0_s2, flow0)
        flow_list.append(flow0)

        feature0_s2 = F.interpolate(feature0_s2, scale_factor=2, mode='nearest')
        feature1_s2 = F.interpolate(feature1_s2, scale_factor=2, mode='nearest')

        # feature0_s1.zero_()
        # feature1_s1.zero_()

        feature0_s1 = self.merge_conv(torch.cat([feature0_s1, feature0_s2], dim=1))
        feature1_s1 = self.merge_conv(torch.cat([feature1_s1, feature1_s2], dim=1))

        flow0 = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2

        feature1_s1 = utils.flow_warp(feature1_s1, flow0)

        delta_flow = self.fine_up_sampler(feature0_s1, feature1_s1, flow0)
        flow0 = flow0 + delta_flow

        flow_list.append(flow0)

        return flow_list
