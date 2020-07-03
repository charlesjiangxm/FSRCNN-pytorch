import math
from torch import nn


class FSRCNN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=1, dilate=32, shrink=5):
        super(FSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, dilate, kernel_size=5, padding=5//2)
        self.prelu1 = nn.PReLU(dilate)

        self.conv2 = nn.Conv2d(dilate, shrink, kernel_size=1)
        self.prelu2 = nn.PReLU(shrink)

        self.conv3 = nn.Conv2d(shrink, shrink, kernel_size=3, padding=3//2)
        self.prelu3 = nn.PReLU(shrink)

        self.conv4 = nn.Conv2d(shrink, dilate, kernel_size=1)
        self.prelu4 = nn.PReLU(dilate)

        self.tconv5 = nn.ConvTranspose2d(dilate, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                         output_padding=scale_factor-1)

        self.conv_in_acts = []
        self.conv_out_acts = []

    def forward(self, x):
        self.conv_in_acts.append(x.data.cpu().numpy()[0])  # 0
        x = self.conv1(x)
        self.conv_out_acts.append(x.data.cpu().numpy()[0])
        x = self.prelu1(x)
        self.conv_in_acts.append(x.data.cpu().numpy()[0])  # 2

        x = self.conv2(x)
        self.conv_out_acts.append(x.data.cpu().numpy()[0])
        x = self.prelu2(x)
        self.conv_in_acts.append(x.data.cpu().numpy()[0])  # 4

        x = self.conv3(x)
        self.conv_out_acts.append(x.data.cpu().numpy()[0])
        x = self.prelu3(x)
        self.conv_in_acts.append(x.data.cpu().numpy()[0])  # 6

        x = self.conv4(x)
        self.conv_out_acts.append(x.data.cpu().numpy()[0])
        x = self.prelu4(x)
        self.conv_in_acts.append(x.data.cpu().numpy()[0])  # 8

        x = self.tconv5(x)
        self.conv_out_acts.append(x.data.cpu().numpy()[0])

        return x
