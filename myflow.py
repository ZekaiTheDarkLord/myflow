import torch
import torch.nn.functional as F

import backbone
import transformer
import matching
import up_sampler_2
import utils

import config
import time


class GMFlow(torch.nn.Module):
    def __init__(self):
        super(GMFlow, self).__init__()
        self.backbone = backbone.CNNEncoder()
        self.cross_attn_2 = transformer.FeatureTransformer(config.feature_dim+2, num_layers=1, bidir=True, ffn=True, ffn_dim_expansion=4, post_norm=True)
        self.cross_attn_2_2 = transformer.FeatureTransformer(config.feature_dim+4, num_layers=2, bidir=True, ffn=True, ffn_dim_expansion=4, post_norm=True)
        
        self.feature_flow_attn = transformer.FeatureFlowAttention(config.feature_dim+6)

        # torch.nn.ConvTranspose2d(config.feature_dim+2, config.feature_dim+2, kernel_size=2, stride=1, bias=True)

        self.merge_conv = torch.nn.Sequential(torch.nn.Conv2d((config.feature_dim+2)*2+6, config.feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.up_sampler = up_sampler_2.UpSampler(config.feature_dim, patch_size=7)

    def init_hw(self, height, width):
        self.backbone.init_hw(height, width)
        # self.up_sampler.init_hw(height, width)

    def forward(self, img_0, img_1):

        flow_list = []

        img_0 = utils.normalize_img(img_0)
        img_1 = utils.normalize_img(img_1)

        feature_0_list = self.backbone(img_0)
        # torch.cuda.synchronize()
        # start_time = time.time()
        feature_1_list = self.backbone(img_1)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('backbone:', end_time-start_time)

        feature_0_1, feature_0_2 = feature_0_list
        feature_1_1, feature_1_2 = feature_1_list

        feature_0_2, feature_1_2 = self.cross_attn_2(feature_0_2, feature_1_2)
        flow = matching.global_correlation_softmax(feature_0_2, feature_1_2, bw_flow=True)
        flow_0, flow_1 = flow.chunk(chunks=2, dim=0)

        feature_0_2 = torch.cat([feature_0_2, flow_0], dim=1)
        feature_1_2 = torch.cat([feature_1_2, flow_1], dim=1)

        feature_0_2, feature_1_2 = self.cross_attn_2_2(feature_0_2, feature_1_2)
        flow = matching.global_correlation_softmax(feature_0_2, feature_1_2, bw_flow=True)
        flow_0, flow_1 = flow.chunk(chunks=2, dim=0)
        
        feature_0_2 = torch.cat([feature_0_2, flow_0], dim=1)
        feature_1_2 = torch.cat([feature_1_2, flow_1], dim=1)

        flow_list.append(flow_0)
        feature_0_2, feature_1_2, flow_0, flow_1 = self.feature_flow_attn(feature_0_2, feature_1_2, flow)
        flow_list.append(flow_0)

        warped_flow_1 = utils.flow_warp(flow_1, flow_0)

        feature_0_2 = torch.cat([feature_0_2, flow_0], dim=1)
        feature_1_2 = torch.cat([feature_1_2, flow_1], dim=1)

        feature_0_2 = F.interpolate(feature_0_2, scale_factor=2, mode='nearest')
        feature_1_2 = F.interpolate(feature_1_2, scale_factor=2, mode='nearest')

        # feature_0_1.zero_()
        # feature_1_1.zero_()

        feature_0_1 = self.merge_conv(torch.cat([feature_0_1, feature_0_2], dim=1))
        feature_1_1 = self.merge_conv(torch.cat([feature_1_1, feature_1_2], dim=1))

        flow_0 = F.interpolate(flow_0, scale_factor=2, mode='nearest') * 2
        flow_1 = F.interpolate(flow_1, scale_factor=2, mode='nearest') * 2

        feature_1_1 = utils.flow_warp(feature_1_1, flow_0)

        warped_flow_1 = F.interpolate(warped_flow_1, scale_factor=2, mode='nearest') * 2

        delta_flow = self.up_sampler(feature_0_1, feature_1_1, flow_0, warped_flow_1)
        flow_0 = flow_0 + delta_flow

        flow_list.append(flow_0)

        return flow_list