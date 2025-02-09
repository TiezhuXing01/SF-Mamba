#!/usr/bin/env python
# -*- coding:utf-8 -*-
# This code partially draws from the implementation of PointFlow ResNet series.
# Original repository: https://github.com/lxtGH/PFSegNets

import torch.nn as nn
import torch
from network.nn.operators import PSPModule
from network.nn.point_flow import PointFlowModuleWithMaxAvgpool
from network.mynet import SemanticBranch, Ablock_v2
from network import resnet_d as Resnet_Deep
from network.nn.mynn import Norm2d, Upsample
import torch.nn.functional as F
from network.vmamba.vmamba import Backbone_VSSM


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )

class UperNetAlignHeadMaxAvgpool(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 fpn_dsn=False, reduce_dim=64, ignore_background=False, max_pool_size=8,
                 avgpool_size=8, edge_points=32, version=1):
        super(UperNetAlignHeadMaxAvgpool, self).__init__()

        self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        
        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
            self.fpn_out_align.append(
                Ablock_v2(fpn_dim, out_channels=reduce_dim, maxpool_size=max_pool_size,
                                              avgpool_size=avgpool_size)
            )

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)      # conv3x3_bn_relu
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)      # A模块

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, fpn_dim//2, kernel_size=1)
        )
        self.conv_last_1 = nn.Sequential(
            conv3x3_bn_relu(768, fpn_dim, 1),
            nn.Conv2d(fpn_dim, fpn_dim//2, kernel_size=1)
        )
        
        self.conv_final = nn.Conv2d(fpn_dim, num_class, 1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 256, 1),
            norm_layer(256),
            nn.ReLU(inplace=False)
        )
        

    def forward(self, conv_out, aux_outs):
        aux_1 = aux_outs[0]   # [b, 96, 128, 128]
        aux_2 = aux_outs[1]   # [b, 192, 64, 64]
        aux_3 = aux_outs[2]   # [b, 384, 32, 32]
        aux_4 = aux_outs[3]   # [b, 768, 16, 16]
        aux_4 = self.conv4(aux_4)   # [b, 2048, 16, 16]
        
        psp_out = self.ppm(conv_out[-1])
        _, _, max_h, max_w = conv_out[0].size()   # 128, 128

        f = psp_out   # (b, 256, 16, 16)

        fpn_feature_list = [f]
        
        out = []
        semantic_branch_list = []

        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            semantic_branch = SemanticBranch(conv_x, aux_4)
            semantic_branch = F.interpolate(semantic_branch, size=(max_h, max_w), mode="bilinear", align_corners=True)
            semantic_branch_list.append(semantic_branch)
            _, _, h, w = conv_x.shape
            f = self.fpn_out_align[i]([f, conv_x])
            f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=True)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear',
                align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        fusion_out_1 = torch.cat(semantic_branch_list, 1)
        x = self.conv_last(fusion_out)

        x_1 = self.conv_last_1(fusion_out_1)

        x = torch.cat((x, x_1), 1)
        out = self.conv_final(x)
        
        return out

class AlignNetResNetMaxAvgpool(nn.Module):
    def __init__(self, num_classes, trunk='resnet-50-deep', criterion=None, variant='D', skip='m1', skip_num=48,
                 fpn_dsn=False, inplanes=128, reduce_dim=64, ignore_background=False,
                 max_pool_size=8, avgpool_size=8, edge_points=32):
        super(AlignNetResNetMaxAvgpool, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num
        self.fpn_dsn = fpn_dsn

        if trunk == trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
        else:
            raise ValueError("Not a valid network arch")

        resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        del resnet
        self.aux_branch = Backbone_VSSM(out_indices=(0, 1, 2, 3), 
                         depths=(2, 2, 8, 2),
                         ssm_conv_bias=False,
                         ssm_ratio=1.0,
                         forward_type="v05_noz", # v3_noz,
                         downsample_version="v1",
                         patchembed_version="v1",
                         mlp_ratio=0,
                         drop_path_rate=0.,
                         norm_layer="ln2d")

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (4, 4), (4, 4)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
        else:
            print("Not using Dilation ")

        inplane_head = 2048
        # inplane_head = 768
        self.head = UperNetAlignHeadMaxAvgpool(inplane_head, num_class=num_classes, norm_layer=Norm2d,
                                               fpn_dsn=fpn_dsn, reduce_dim=reduce_dim,
                                               ignore_background=ignore_background, max_pool_size=max_pool_size,
                                               avgpool_size=avgpool_size, edge_points=edge_points)

    def forward(self, x, gts=None):
        x0 = self.layer0(x)    # [b, 128, 128, 128]
        x1 = self.layer1(x0)   # [b, 256, 128, 128]
        x2 = self.layer2(x1)   # [b, 512, 64, 64]
        x3 = self.layer3(x2)   # [b, 1024, 32, 32]
        x4 = self.layer4(x3)   # [b, 2048, 16, 16]
        aux_outs = self.aux_branch(x)
        out = self.head([x1, x2, x3, x4], aux_outs)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
        if self.training:
            return self.criterion(out, gts)
        return out


def DeepR101_PF_maxavg_deeply(num_classes, criterion, reduce_dim=64, max_pool_size=8, avgpool_size=8, edge_points=32):
    """
    ResNet-101 Based Network
    """
    return AlignNetResNetMaxAvgpool(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', skip='m1',
                                    reduce_dim=reduce_dim, max_pool_size=max_pool_size, avgpool_size=avgpool_size,
                                    edge_points=edge_points)

def DeepR50_PF_maxavg_deeply(num_classes, criterion, reduce_dim=64, max_pool_size=8, avgpool_size=8, edge_points=32):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNetMaxAvgpool(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1',
                                    reduce_dim=reduce_dim, max_pool_size=max_pool_size, avgpool_size=avgpool_size,
                                    edge_points=edge_points)
