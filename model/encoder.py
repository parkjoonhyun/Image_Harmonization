import torch
from torch import nn
import torch.nn.functional as F
from model.modelUtils import pad_tensor , pad_tensor_back , Normalization , Pooling , h_sigmoid , SELayer , Conv2dBlock

class ReferenceEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(ReferenceEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        # self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.se = SELayer(512)
        self.output_dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.model(x)
        out = self.se(x)
        reference = self.avg_pool(out)
        return reference


class ContentEncoder(nn.Module):
    def __init__(self, opt):
        super(ContentEncoder, self).__init__()
        self.opt = opt
        p = 1
        # self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = Normalization(opt, 32)

        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = Normalization(opt, 32)
        self.max_pool1 = Pooling(opt, 2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = Normalization(opt, 64)

        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = Normalization(opt, 64)
        self.max_pool2 = Pooling(opt, 2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = Normalization(opt, 128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = Normalization(opt, 128)
        self.max_pool3 = Pooling(opt, 2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = Normalization(opt, 256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = Normalization(opt, 256)
        self.max_pool4 = Pooling(opt, 2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_1 = Normalization(opt, 512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_2 = Normalization(opt, 512)

        self.se_1 = SELayer(512)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.content_style_conv = nn.Conv2d(1024, 512, 3, padding=p)

        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_1 = Normalization(opt, 256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_2 = Normalization(opt, 256)

        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = Normalization(opt, 128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = Normalization(opt, 128)

        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = Normalization(opt, 64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = Normalization(opt, 64)

        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = Normalization(opt, 32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 3, 1)

    def forward(self, input):
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)

        conv =[]

        x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        conv.append(conv1)
        x = self.max_pool1(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        conv.append(conv2)
        x = self.max_pool2(conv2)

        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        conv.append(conv3)
        x = self.max_pool3(conv3)

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
        conv.append(conv4)
        x = self.max_pool4(conv4)

        x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
        h = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))
        content = self.se_1(h)
        return content , conv , pad_left , pad_right , pad_top , pad_bottom
