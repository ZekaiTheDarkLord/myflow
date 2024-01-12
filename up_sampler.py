import torch
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler


class UpSampler(torch.nn.Module):
    def __init__(self, feature_dim, patch_size):
        super(UpSampler, self).__init__()

        self.patch_size = patch_size

        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=patch_size, stride=1, padding=0, dilation=1)

        # self.attn_conv = torch.nn.Conv2d(in_channels=patch_size**2*3, out_channels=patch_size**2*2, kernel_size=1, stride=1, padding=0)

        self.conv1 = torch.nn.Conv2d(in_channels=patch_size**2+feature_dim+4, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def init_hw(self, height, width):

        self.local_ys, self.local_xs = torch.meshgrid(torch.arange(self.patch_size), torch.arange(self.patch_size), indexing='ij')

        self.local_ys = self.local_ys - (self.patch_size - 1) // 2
        self.local_xs = self.local_xs - (self.patch_size - 1) // 2

        self.local_ys = self.local_ys.flatten().cuda()[None,:,None,None]
        self.local_xs = self.local_xs.flatten().cuda()[None,:,None,None]

        self.local_grid = torch.cat([self.local_xs, self.local_ys], dim=1)
        self.local_grid = self.local_grid.repeat(1, 1, height, width)

    def forward(self, feature_0, feature_1, flow_0, flow_1):

        b, c, h, w = feature_0.shape

        attn = self.correlation_sampler(feature_0, feature_1).view(b, -1, h, w)
        attn = F.softmax(attn, dim=1)

        x = torch.cat([attn, feature_0, flow_0, flow_1], dim=1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        return self.conv4(x)