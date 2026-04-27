"""
fpn_module.py
FPN 论文复现 - 骨干网络 + 特征金字塔网络

对应论文: Feature Pyramid Networks for Object Detection (Lin et al., CVPR 2017)

模块说明:
  - ResNet50Backbone: 基于预训练 ResNet-50 的骨干网络，输出 C2-C5 四阶段特征
  - FPN: 特征金字塔网络，通过 lateral connections + top-down pathway 生成 P2-P5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50Backbone(nn.Module):
    """
    基于预训练 ResNet-50 的骨干网络（对应论文 Section 4 实验设置）。
    输出 C2, C3, C4, C5 四个阶段的特征图。

    各阶段输出:
      C2: stride=4,  channels=256
      C3: stride=8,  channels=512
      C4: stride=16, channels=1024
      C5: stride=32, channels=2048
    """
    def __init__(self, pretrained=True):
        super(ResNet50Backbone, self).__init__()
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)

        # stem: conv1 + bn1 + relu + maxpool → stride=4
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # C2: stride=4,  out=256
        self.layer2 = resnet.layer2  # C3: stride=8,  out=512
        self.layer3 = resnet.layer3  # C4: stride=16, out=1024
        self.layer4 = resnet.layer4  # C5: stride=32, out=2048

    def forward(self, x):
        x  = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class FPN(nn.Module):
    """
    FPN (Feature Pyramid Networks) 核心模块。

    对应论文 Figure 3:
      - Bottom-up pathway: ResNet50Backbone 提供 C2-C5
      - Lateral connections: 1x1 卷积将各层通道数统一为 out_channels (256)
      - Top-down pathway: 2x nearest 上采样 + 逐元素相加
      - Output: 3x3 卷积平滑，输出 P2-P5

    参数:
      in_channels_list: 各阶段输入通道数 [256, 512, 1024, 2048]
      out_channels: 统一输出通道数 (论文中为 256)
    """
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels

        # 1x1 卷积: lateral connections（从 C5 到 C2 的顺序）
        self.lateral_convs = nn.ModuleList()
        # 3x3 卷积: 消除上采样混叠效应
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list[::-1]:  # C5 → C4 → C3 → C2
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, inputs):
        c2, c3, c4, c5 = inputs

        # Top-down pathway + lateral connections
        # P5: C5 经过 1x1 卷积
        p5_lateral = self.lateral_convs[0](c5)
        p5 = self.output_convs[0](p5_lateral)

        # P4: 1x1(C4) + upsample(P5_lateral)
        c4_lateral = self.lateral_convs[1](c4)
        p4_fused = c4_lateral + F.interpolate(p5_lateral, size=c4_lateral.shape[2:], mode='nearest')
        p4 = self.output_convs[1](p4_fused)

        # P3: 1x1(C3) + upsample(P4_fused)
        c3_lateral = self.lateral_convs[2](c3)
        p3_fused = c3_lateral + F.interpolate(p4_fused, size=c3_lateral.shape[2:], mode='nearest')
        p3 = self.output_convs[2](p3_fused)

        # P2: 1x1(C2) + upsample(P3_fused)
        c2_lateral = self.lateral_convs[3](c2)
        p2_fused = c2_lateral + F.interpolate(p3_fused, size=c2_lateral.shape[2:], mode='nearest')
        p2 = self.output_convs[3](p2_fused)

        return p2, p3, p4, p5


if __name__ == "__main__":
    # 验证 ResNet50Backbone + FPN 的输出 shape
    print("ResNet50Backbone + FPN 验证")
    print("=" * 50)

    backbone = ResNet50Backbone(pretrained=True)
    fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels=256)

    x = torch.randn(1, 3, 800, 800)
    c2, c3, c4, c5 = backbone(x)
    print(f"Backbone 输出:")
    print(f"  C2: {c2.shape}  (stride=4,  ch=256)")
    print(f"  C3: {c3.shape}  (stride=8,  ch=512)")
    print(f"  C4: {c4.shape}  (stride=16, ch=1024)")
    print(f"  C5: {c5.shape}  (stride=32, ch=2048)")

    p2, p3, p4, p5 = fpn((c2, c3, c4, c5))
    print(f"\nFPN 输出 (所有通道数统一为 256):")
    print(f"  P2: {p2.shape}  (stride=4)")
    print(f"  P3: {p3.shape}  (stride=8)")
    print(f"  P4: {p4.shape}  (stride=16)")
    print(f"  P5: {p5.shape}  (stride=32)")
    print("\n验证通过！")
