import torch
import torch.nn as nn
import torch.nn.functional as F
use_cuda=torch.cuda.is_available()

class SeperableConv2d(nn.Module):

    # ***Figure 4. An “extreme” version of our Inception module,
    # with one spatial convolution per output channel of the 1x1
    # convolution."""
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels,
            bias=False,
            **kwargs
        )

        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class EntryFlow(nn.Module):

    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3_residual = nn.Sequential(
            SeperableConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )

        self.conv4_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2),
            nn.BatchNorm2d(256),
        )

        # no downsampling
        self.conv5_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, 1, padding=1)
        )

        # no downsampling
        self.conv5_shortcut = nn.Sequential(
            nn.Conv2d(256, 728, 1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual + shortcut
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x = residual + shortcut

        return x


class MiddleFLowBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        shortcut = self.shortcut(x)

        return shortcut + residual


class MiddleFlow(nn.Module):
    def __init__(self, block):
        super().__init__()

        # """then through the middle flow which is repeated eight times"""
        self.middel_block = self._make_flow(block, 8)

    def forward(self, x):
        x = self.middel_block(x)
        return x

    def _make_flow(self, block, times):
        flows = []
        for i in range(times):
            flows.append(block())

        return nn.Sequential(*flows)


class ExitFLow(nn.Module):

    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeperableConv2d(728, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=2),
            nn.BatchNorm2d(1024)
        )

        self.conv = nn.Sequential(
            SeperableConv2d(1024, 1536, 3, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeperableConv2d(1536, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output = shortcut + residual
        output = self.conv(output)
        # output = self.avgpool(output)

        return output


class Xception(nn.Module):

    def __init__(self, block=MiddleFLowBlock, num_classes=4, input_channels=1):
        super().__init__()
        self.entry_flow = EntryFlow(input_channels=input_channels)
        self.middel_flow = MiddleFlow(block)
        self.exit_flow = ExitFLow()
        # self.fc = nn.Linear(2048, num_class)

        self.convt0 = nn.Conv2d(2048, num_classes, 1)
        self.convt1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.pos_output = nn.Conv2d(num_classes, 1, kernel_size=1)
        self.cos_output = nn.Conv2d(num_classes, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(num_classes, 1, kernel_size=1)
        self.width_output = nn.Conv2d(num_classes, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.05)



    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middel_flow(x)
        x = self.exit_flow(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x = self.convt0(x)
        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)
        x = self.dropout(x)

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        # return x
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


def xception():
    model=Xception(MiddleFLowBlock)
    if use_cuda:
        model=model.cuda()
    return model
