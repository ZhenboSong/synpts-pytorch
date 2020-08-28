import torch
import torch.nn as nn
from models.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            # elif classname.find('BatchNorm2d') != -1:
                # nn.init.normal_(m.weight.data, 1.0, gain)
                # nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class PtEncoder(BaseNetwork):
    def __init__(self, cfg, init_weights=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels = cfg.points_channel
        use_xyz = cfg.use_xyz

        skip_channel_list = [input_channels]
        for k in range(cfg.sa_npoints.__len__()):
            mlps = cfg.sa_mlps[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=cfg.sa_npoints[k],
                    radii=cfg.sa_radius[k],
                    nsamples=cfg.sa_nsample[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.use_bn
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(cfg.fp_mlps.__len__()):
            pre_channel = cfg.fp_mlps[k + 1][-1] if k + 1 < len(cfg.fp_mlps) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + cfg.fp_mlps[k])
            )
        if init_weights:
            self.init_weights('xavier')

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features, l_xyz


class UnetUp(nn.Module):
    def __init__(self, x1_ch, x2_ch, out_ch, bilinear=True):
        super(UnetUp, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(x1_ch + x2_ch, out_ch, 3, padding=0),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3, padding=0),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(x1_ch, int(x1_ch/2), 2, stride=2)
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(int(x1_ch/2) + x2_ch, out_ch, 3, padding=0),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3, padding=0),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UnetDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnetDown, self).__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=0),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, padding=0),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ImgDecoder(BaseNetwork):
    def __init__(self, init_weights=True):
        super(ImgDecoder, self).__init__()
        # feature_chs = [1024, 512, 512, 256, 128]
        # out_chs = [512, 256, 128, 64, 3]
        self.up4 = UnetUp(1024, 512, 512)
        self.up3 = UnetUp(512, 512, 256)
        self.up2 = UnetUp(256, 256, 128)
        self.up1 = UnetUp(128, 128, 64)

        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        # x = [fm0, fm1, fm2, fm4, fm5] from high resolution to low resolution
        fm1, fm2, fm3, fm4, fm5 = x
        img = self.up4(fm5, fm4)
        img = self.up3(img, fm3)
        img = self.up2(img, fm2)
        img = self.up1(img, fm1)
        img = self.decoder(img)
        img = (torch.tanh(img) + 1) / 2
        return img


class RefineGenerator(BaseNetwork):
    def __init__(self, in_ch=3, out_ch=3, init_weights=True):
        super(RefineGenerator, self).__init__()

        self.inc = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, 64, 3, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.down1 = UnetDown(64, 128)
        self.down2 = UnetDown(128, 256)
        self.down3 = UnetDown(256, 512)
        self.down4 = UnetDown(512, 512)
        self.up1 = UnetUp(512, 512, 256)
        self.up2 = UnetUp(256, 256, 128)
        self.up3 = UnetUp(128, 128, 64)
        self.up4 = UnetUp(64, 64, 64)

        self.outc = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, out_ch, 3, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
