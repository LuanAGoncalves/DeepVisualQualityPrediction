import torch
import torch.nn as nn
from torch.nn import functional as F


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(
            inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class Dense_base_down2(nn.Module):
    def __init__(self):
        super(Dense_base_down2, self).__init__()

        self.dense_block1 = BottleneckBlock(1, 13)
        self.trans_block1 = TransitionBlock1(14, 8)

        ############# Block2-down 32-32  ##############
        self.dense_block2 = BottleneckBlock(8, 16)
        self.trans_block2 = TransitionBlock1(24, 16)

        ############# Block3-down  16-16 ##############
        self.dense_block3 = BottleneckBlock(16, 16)
        self.trans_block3 = TransitionBlock3(32, 16)

        ############# Block4-up  8-8  ##############
        self.dense_block4 = BottleneckBlock(16, 16)
        self.trans_block4 = TransitionBlock3(32, 16)

        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckBlock(16, 8)
        self.trans_block5 = TransitionBlock(24, 8)

        self.dense_block6 = BottleneckBlock(8, 8)
        self.trans_block6 = TransitionBlock(16, 2)

        self.conv_refin = nn.Conv2d(11, 20, 3, 1, 1)
        self.tanh = nn.Tanh()

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm

        self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.batchnorm20 = nn.BatchNorm2d(20)
        self.batchnorm1 = nn.BatchNorm2d(1)

    def forward(self, x):
        ## 256x256
        x1 = self.dense_block1(x)
        x1 = self.trans_block1(x1)

        ###  32x32
        x2 = self.dense_block2(x1)
        x2 = self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3 = self.dense_block3(x2)
        x3 = self.trans_block3(x3)

        ## Classifier  ##

        x4 = self.dense_block4(x3)
        x4 = self.trans_block4(x4)

        x4 = x4 + x2

        x5 = self.dense_block5(x4)
        x5 = self.trans_block5(x5)

        x5 = x5 + x1

        x6 = self.dense_block6(x5)
        x6 = self.trans_block6(x6)

        return x6


class Dense_base_down1(nn.Module):
    def __init__(self):
        super(Dense_base_down1, self).__init__()

        self.dense_block1 = BottleneckBlock(1, 13)
        self.trans_block1 = TransitionBlock1(14, 8)

        ############# Block2-down 32-32  ##############
        self.dense_block2 = BottleneckBlock(8, 16)
        self.trans_block2 = TransitionBlock3(24, 16)

        ############# Block3-down  16-16 ##############
        self.dense_block3 = BottleneckBlock(16, 16)
        self.trans_block3 = TransitionBlock3(32, 16)

        ############# Block4-up  8-8  ##############
        self.dense_block4 = BottleneckBlock(16, 16)
        self.trans_block4 = TransitionBlock3(32, 16)

        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckBlock(16, 8)
        self.trans_block5 = TransitionBlock3(24, 8)

        self.dense_block6 = BottleneckBlock(8, 8)
        self.trans_block6 = TransitionBlock(16, 2)

        self.conv_refin = nn.Conv2d(11, 20, 3, 1, 1)
        self.tanh = nn.Tanh()

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm

        self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.batchnorm20 = nn.BatchNorm2d(20)
        self.batchnorm1 = nn.BatchNorm2d(1)

    def forward(self, x):
        ## 256x256
        x1 = self.dense_block1(x)
        x1 = self.trans_block1(x1)

        ###  32x32
        x2 = self.dense_block2(x1)
        x2 = self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3 = self.dense_block3(x2)
        x3 = self.trans_block3(x3)

        ## Classifier  ##

        x4 = self.dense_block4(x3)
        x4 = self.trans_block4(x4)

        x4 = x4 + x2

        x5 = self.dense_block5(x4)
        x5 = self.trans_block5(x5)

        x5 = x5 + x1

        x6 = self.dense_block6(x5)
        x6 = self.trans_block6(x6)

        return x6


class Dense_base_down0(nn.Module):
    def __init__(self):
        super(Dense_base_down0, self).__init__()

        self.dense_block1 = BottleneckBlock(1, 5)
        self.trans_block1 = TransitionBlock3(6, 4)

        ############# Block2-down 32-32  ##############
        self.dense_block2 = BottleneckBlock(4, 8)
        self.trans_block2 = TransitionBlock3(12, 12)

        ############# Block3-down  16-16 ##############
        self.dense_block3 = BottleneckBlock(12, 4)
        self.trans_block3 = TransitionBlock3(16, 12)

        ############# Block4-up  8-8  ##############
        self.dense_block4 = BottleneckBlock(12, 4)
        self.trans_block4 = TransitionBlock3(16, 12)

        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckBlock(12, 8)
        self.trans_block5 = TransitionBlock3(20, 4)

        self.dense_block6 = BottleneckBlock(4, 8)
        self.trans_block6 = TransitionBlock3(12, 2)

    def forward(self, x):
        ## 256x256
        x1 = self.dense_block1(x)
        x1 = self.trans_block1(x1)

        ###  32x32
        x2 = self.dense_block2(x1)
        x2 = self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3 = self.dense_block3(x2)
        x3 = self.trans_block3(x3)

        ## Classifier  ##

        x4 = self.dense_block4(x3)
        x4 = self.trans_block4(x4)

        x4 = x4 + x2

        x5 = self.dense_block5(x4)
        x5 = self.trans_block5(x5)

        x5 = x5 + x1

        x6 = self.dense_block6(x5)
        x6 = self.trans_block6(x6)

        return x6


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)


class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class MultiscaleDSP(nn.Module):
    def __init__(self):
        super(MultiscaleDSP, self).__init__()
        self.dense0 = Dense_base_down0()
        self.dense1 = Dense_base_down1()
        self.dense2 = Dense_base_down2()

        self.conv1 = nn.Conv2d(6, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 24, 3, 1, 1)
        self.fc1 = nn.Linear(24576, 512)
        self.bnfc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x3 = self.dense2(x)
        x2 = self.dense1(x)
        x1 = self.dense0(x)

        output = torch.cat([x1, x2, x3], 1)

        output = self.relu(self.bn1(self.conv1(output)))
        output = self.relu(self.bn2(self.conv2(output)))

        n, c, h, w = output.shape

        output = output.view(n, c * h * w)

        output = self.relu(self.bnfc(self.fc1(output)))
        output = self.relu(self.fc2(output))

        return output


class Default(nn.Module):
    def __init__(self):
        super(Default, self).__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.features(x)
        n, c, _, _ = output.shape
        output = output.view(n, c)
        output = self.regressor(output)
        return output


if __name__ == "__main__":
    x = torch.rand(32, 1, 32, 32)
    net = Default()
    print(net(x))
