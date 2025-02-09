import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureMatcher(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(FeatureMatcher, self).__init__()
        self.match_conv = nn.Conv2d(dim*2, 1, kernel_size, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_high, x_low = x
        x_low = F.upsample(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        certainty = self.match_conv(torch.cat([x_high, x_low], dim=1))
        return self.sigmoid(certainty)

def SemanticBranch(conv_x, psp_out):
    b, c, h, w = conv_x.size()    # psp_out (b, 256, h, w)

    psp_out = F.interpolate(psp_out, size=(h, w), mode="bilinear", align_corners=True)

    similarity_matrix = (conv_x * psp_out).sum()    # (b, 1, h, w)
    
    similarity = torch.sigmoid(similarity_matrix)  # (b, h*w, h*w)
    
    output = conv_x * similarity
    
    return output
    
class Ablock_v2(nn.Module):
    def __init__(self, in_channels, out_channels=64, maxpool_size=8, avgpool_size=8, matcher_kernel_size=3):
        super(Ablock_v2, self).__init__()
        self.maxpool_size = maxpool_size
        self.avgpool_size = avgpool_size
        self.max_pool = nn.AdaptiveMaxPool2d((maxpool_size, maxpool_size))
        self.avg_pool = nn.AdaptiveAvgPool2d((avgpool_size, avgpool_size))
        self.point_matcher = FeatureMatcher(out_channels, matcher_kernel_size)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_2 = nn.Conv2d(512, 256, 1)


    def forward(self, x):
        x_high, x_low = x
        x_high_embed = self.conv_1(x_high)
        x_low_embed = self.conv_1(x_low)

        certainty_map = self.point_matcher([x_high_embed, x_low_embed])
        avgpool_grid = self.avg_pool(certainty_map)
        _, _, map_h, map_w = certainty_map.size()
        avgpool_grid = F.interpolate(avgpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)
        feature_a = x_high - x_high * avgpool_grid

        maxpool_grid = self.max_pool(certainty_map)
        maxpool_grid = F.interpolate(maxpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)
        feature_b = x_high + x_high * maxpool_grid

        f = torch.cat((feature_a, feature_b), 1)
        f = self.conv_2(f)

        return f