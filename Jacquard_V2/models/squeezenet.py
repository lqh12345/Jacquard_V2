import torch
import torch.nn as nn
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()

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

class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )
        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)
        return x
class SqueezeNet(nn.Module):

    def __init__(self, input_channels=1, class_num=4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.maxpool = nn.MaxPool2d(2, 2)

        self.fire2 = Fire(64, 32, 8)
        self.fire3 = Fire(32, 32, 8)
        self.fire4 = Fire(32, 16, 32)
        self.fire5 = Fire(16, 16, 32)
        self.fire6 = Fire(16, 32, 24)
        self.fire7 = Fire(32, 32, 24)
        self.fire8 = Fire(32, 64, 32)
        self.fire9 = Fire(64, 64, 32)
        self.conv10 = nn.Conv2d(64, class_num, 1)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()
        # self.upsample1 = nn.Upsample(size=300, mode='nearest')
        self.maxpool = nn.MaxPool2d(2, 2)
        # self.maxpool = nn.MaxPool2d((2, 2), stride=(1, 1))
        #, ceil_mode=True
        # self.upsample2 = nn.Upsample(size=300, mode='nearest')
        # torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)
        self.convt1 = nn.ConvTranspose2d(class_num, class_num, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(class_num, class_num, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(class_num, class_num, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.pos_output = nn.Conv2d(class_num, 1, kernel_size=1)
        self.cos_output = nn.Conv2d(class_num, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(class_num, 1, kernel_size=1)
        self.width_output = nn.Conv2d(class_num, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.maxpool(x)
        x = self.fire2(x)
        x = self.fire3(x) + x
        x = self.fire4(x)
        x = self.maxpool(x)

        x = self.fire5(x) + x
        x = self.fire6(x)
        x = self.fire7(x) + x
        x = self.fire8(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        #x = self.upsample1(x)
        x = self.maxpool(x)
        x = self.fire9(x)
        x = self.conv10(x)
        #x = self.upsample2(x)

        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)
        return pos_output, cos_output, sin_output, width_output


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





