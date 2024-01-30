from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
    
    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

class DepthwiseConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        return self.depthwise(x)

class PointwiseConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, **kwargs),
            nn.BatchNorm2d(output_channels)
        )
    
    def forward(self, x):
        return self.pointwise(x)

class ShuffleNetUnit(nn.Module):

    def __init__(self, input_channels, output_channels, stage, stride, groups):
        super().__init__()

        self.gp1 = nn.Sequential(
            PointwiseConv2d(
                input_channels, 
                int(output_channels / 4), 
                groups=groups
            ),
            nn.ReLU(inplace=True)
        )

        
        self.channel_shuffle = ChannelShuffle(groups)

        self.depthwise = DepthwiseConv2d(
            int(output_channels / 4), 
            int(output_channels / 4), 
            3, 
            groups=int(output_channels / 4), 
            stride=stride,
            padding=1
        )

        self.expand = PointwiseConv2d(
            int(output_channels / 4),
            output_channels,
            groups=groups
        )

        self.relu = nn.ReLU(inplace=True)
        self.fusion = self._add
        self.shortcut = nn.Sequential()

        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)

            self.expand = PointwiseConv2d(
                int(output_channels / 4),
                output_channels - input_channels,
                groups=groups
            )

            self.fusion = self._cat
    
    def _add(self, x, y):
        return torch.add(x, y)
    
    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        shortcut = self.shortcut(x)

        shuffled = self.gp1(x)
        shuffled = self.channel_shuffle(shuffled)
        shuffled = self.depthwise(shuffled)
        shuffled = self.expand(shuffled)

        output = self.fusion(shortcut, shuffled)
        output = self.relu(output)

        return output

class ShuffleNet(nn.Module):

    def __init__(self, num_blocks=[4, 8, 4], input_channels=1, class_num=4, groups=3):
        super().__init__()

        if groups == 1:
            out_channels = [24, 144, 288, 567]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [18, 36, 72, 108]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]

        self.conv1 = BasicConv2d(input_channels, out_channels[0], 3, padding=1, stride=1)
        self.ca = ChannelAttention(out_channels[0])
        self.sa = SpatialAttention()
        self.input_channels = out_channels[0]

        self.stage2 = self._make_stage(
            ShuffleNetUnit, 
            num_blocks[0], 
            out_channels[1], 
            stride=2, 
            stage=2,
            groups=groups
        )

        self.stage3 = self._make_stage(
            ShuffleNetUnit, 
            num_blocks[1], 
            out_channels[2], 
            stride=2,
            stage=3, 
            groups=groups
        )

        self.stage4 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[2],
            out_channels[3],
            stride=2,
            stage=4,
            groups=groups
        )
        self.conv10 = nn.Conv2d(out_channels[3], class_num, 1)
        self.ca1 = ChannelAttention(out_channels[3])
        self.sa1 = SpatialAttention()
        self.convt1 = nn.ConvTranspose2d(class_num, class_num, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(class_num, class_num, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(class_num, class_num, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.pos_output = nn.Conv2d(class_num, 1, kernel_size=1)
        self.cos_output = nn.Conv2d(class_num, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(class_num, 1, kernel_size=1)
        self.width_output = nn.Conv2d(class_num, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv10(x)

        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)
        return pos_output, cos_output, sin_output, width_output

    def _make_stage(self, block, num_blocks, output_channels, stride, stage, groups):

        strides = [stride] + [1] * (num_blocks - 1)

        stage = []

        for stride in strides:
            stage.append(
                block(
                    self.input_channels, 
                    output_channels, 
                    stride=stride, 
                    stage=stage, 
                    groups=groups
                )
            )
            self.input_channels = output_channels

        return nn.Sequential(*stage)
    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
