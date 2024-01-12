import torch
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler


class UpSampler(torch.nn.Module):
    def __init__(self, feature_dim, patch_size):
        super(UpSampler, self).__init__()

        self.patch_size = patch_size

        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=patch_size, stride=1, padding=0, dilation=1)

        self.conv1 = torch.nn.Conv2d(in_channels=patch_size**2+feature_dim+4, out_channels=96, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)

        self.conv5 = torch.nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, feature_0, feature_1, flow_0, warped_flow_1):

        b, c, h, w = feature_0.shape

        attn = self.correlation_sampler(feature_0, feature_1).view(b, -1, h, w)
        attn = F.softmax(attn, dim=1)

        x = torch.cat([attn, feature_0, flow_0, warped_flow_1], dim=1)

        x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))

        return self.conv7(x)