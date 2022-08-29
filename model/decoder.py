import torch
from torch import nn
import torch.nn.functional as F
from model.modelUtils import pad_tensor , pad_tensor_back , Normalization , Pooling , h_sigmoid , SELayer , Conv2dBlock

# Normalization Opt 로 설정 변경 가능하게 롤백해야함

class Decoder(nn.Module):
    def __init__(self , opt):
        super(Decoder , self).__init__()

        p = 1

        self.opt = opt
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

    def forward(self , content , reference , conv , pad_left, pad_right, pad_top, pad_bottom ):
        content = content
        style = reference
        conv = conv

        pad_left = pad_left
        pad_right = pad_right
        pad_top = pad_top
        pad_bottom = pad_bottom

        _, _, h, w = content.size()
        ref_h = style.repeat(1, 1, h, w).view(-1, 512, h, w)
        content_style_cat = torch.cat([content, ref_h], 1)

        content_style_cat = self.content_style_conv(content_style_cat)
        content_style_cat = F.upsample(content_style_cat, scale_factor=2, mode='bilinear')
        up6 = torch.cat([self.deconv5(content_style_cat), conv[3]], 1)
        x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
        conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

        conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
        up7 = torch.cat([self.deconv6(conv6), conv[2]], 1)
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

        conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
        up8 = torch.cat([self.deconv7(conv7), conv[1]], 1)
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

        conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
        up9 = torch.cat([self.deconv8(conv8), conv[0]], 1)
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))

        latent = self.conv10(conv9)

        output = latent
        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        return output


