import torch
import torch.nn as nn


def select_net(config, data_shape):
    return eval(config['name'])(config, data_shape)


# To debug
class PrintLayer(nn.Module):
    def __init__(self, name):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        print(self.name + ': -> ', x.shape)
        return x


class Encoder64x64Baseline(nn.Module):
    def __init__(self, config, data_shape):
        super(Encoder64x64Baseline, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        latent_size = config['latent_size']
        self.main = nn.Sequential(
            # input is (number_bands) x 64 x 64
            nn.Conv2d(number_bands, filter_scale, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale) x 32 x 32
            nn.Conv2d(filter_scale, filter_scale * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filter_scale * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale*2) x 16 x 16
            nn.Conv2d(filter_scale * 2, filter_scale * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filter_scale * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale*4) x 8 x 8
            nn.Conv2d(filter_scale * 4, filter_scale * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filter_scale * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # state size (filter_scale*8) x 4 x 4
        )
        self.conv_mean = nn.Conv2d(filter_scale * 8, latent_size, kernel_size=4, stride=1, padding=0)
        self.conv_logvar = nn.Conv2d(filter_scale * 8, latent_size, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        x = self.main(x)
        return torch.flatten(self.conv_mean(x), 1), torch.flatten(self.conv_logvar(x), 1)


class Encoder64x64ThreeBlocks(nn.Module):
    def __init__(self, config, data_shape):
        super(Encoder64x64ThreeBlocks, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        latent_size = config['latent_size']
        self.main = nn.Sequential(
            # input is (number_bands) x 64 x 64
            nn.Conv2d(number_bands, filter_scale, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale) x 32 x 32
            nn.Conv2d(filter_scale, filter_scale * 2, kernel_size=4, stride=3, padding=1),
            nn.BatchNorm2d(filter_scale * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale*2) x 11 x 11
            nn.Conv2d(filter_scale * 2, filter_scale * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filter_scale * 4),
            nn.LeakyReLU(0.2, inplace=True)
            # state size (filter_scale*4) x 5 x 5
        )
        self.conv_mean = nn.Conv2d(filter_scale * 4, latent_size, kernel_size=5, stride=1, padding=0)
        self.conv_logvar = nn.Conv2d(filter_scale * 4, latent_size, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        x = self.main(x)
        return torch.flatten(self.conv_mean(x), 1), torch.flatten(self.conv_logvar(x), 1)


class Encoder64x64TwoBlocks(nn.Module):
    def __init__(self, config, data_shape):
        super(Encoder64x64TwoBlocks, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        latent_size = config['latent_size']
        self.main = nn.Sequential(
            # input is (number_bands) x 64 x 64
            nn.Conv2d(number_bands, filter_scale, kernel_size=4, stride=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale) x 32 x 32
            nn.Conv2d(filter_scale, filter_scale * 2, kernel_size=4, stride=3, padding=1),
            nn.BatchNorm2d(filter_scale * 2),
            nn.LeakyReLU(0.2, inplace=True)
            # state size (filter_scale*4) x 7 x 7
        )
        self.conv_mean = nn.Conv2d(filter_scale * 2, latent_size, kernel_size=7, stride=1, padding=0)
        self.conv_logvar = nn.Conv2d(filter_scale * 2, latent_size, kernel_size=7, stride=1, padding=0)

    def forward(self, x):
        x = self.main(x)
        return torch.flatten(self.conv_mean(x), 1), torch.flatten(self.conv_logvar(x), 1)


class Decoder64x64Baseline(nn.Module):
    def __init__(self, config, data_shape):
        super(Decoder64x64Baseline, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        self.latent_size = config['latent_size']
        self.filter_scale = filter_scale
        self.main = nn.Sequential(
            # input is (latent_size) x 1 x 1
            nn.ConvTranspose2d(self.latent_size, filter_scale * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(filter_scale * 8),
            nn.ReLU(True),
            # state size (filter_scale*8) x 4 x 4
            nn.ConvTranspose2d(filter_scale * 8, filter_scale * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filter_scale * 4),
            nn.ReLU(True),
            # state size (filter_scale*4) x 8 x 8
            nn.ConvTranspose2d(filter_scale * 4, filter_scale * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filter_scale * 2),
            nn.ReLU(True),
            # state size (filter_scale*2) x 16 x 16
            nn.ConvTranspose2d(filter_scale * 2, filter_scale, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filter_scale),
            nn.ReLU(True),
            # state size (filter_scale) x 32 x 32
            nn.ConvTranspose2d(filter_scale, number_bands, kernel_size=4, stride=2, padding=1),
            # state size (number_bands) x 64 x 64
            nn.Sigmoid()
        )

    def forward(self, x, batch_size):
        return self.main(x.view(batch_size, self.latent_size, 1, 1))


class Decoder64x64FourBlocks(nn.Module):
    def __init__(self, config, data_shape):
        super(Decoder64x64FourBlocks, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        self.latent_size = config['latent_size']
        self.filter_scale = filter_scale
        self.main = nn.Sequential(
            # input is (latent_size) x 1 x 1
            nn.ConvTranspose2d(self.latent_size, filter_scale * 4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(filter_scale * 4),
            nn.ReLU(True),
            # state size (filter_scale*8) x 4 x 4
            nn.ConvTranspose2d(filter_scale * 4, filter_scale * 2, kernel_size=4, stride=3, padding=1),
            nn.BatchNorm2d(filter_scale * 2),
            nn.ReLU(True),
            # state size (filter_scale*4) x 11 x 11
            nn.ConvTranspose2d(filter_scale * 2, filter_scale, kernel_size=4, stride=3, padding=1),
            nn.BatchNorm2d(filter_scale),
            nn.ReLU(True),
            # state size (filter_scale) x 32 x 32
            nn.ConvTranspose2d(filter_scale, number_bands, kernel_size=4, stride=2, padding=1),
            # state size (number_bands) x 64 x 64
            nn.Sigmoid()
        )

    def forward(self, x, batch_size):
        return self.main(x.view(batch_size, self.latent_size, 1, 1))


class Decoder64x64ThreeBlocks(nn.Module):
    def __init__(self, config, data_shape):
        super(Decoder64x64ThreeBlocks, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        self.latent_size = config['latent_size']
        self.filter_scale = filter_scale
        self.main = nn.Sequential(
            # input is (latent_size) x 1 x 1
            nn.ConvTranspose2d(self.latent_size, filter_scale * 2, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(filter_scale * 2),
            nn.ReLU(True),
            # state size (filter_scale*2) x 4 x 4
            nn.ConvTranspose2d(filter_scale * 2, filter_scale, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(filter_scale),
            nn.ReLU(True),
            # state size (filter_scale) x 32 x 32
            nn.ConvTranspose2d(filter_scale, number_bands, kernel_size=6, stride=3, padding=1),
            # state size (number_bands) x 64 x 64
            nn.Sigmoid()
        )

    def forward(self, x, batch_size):
        return self.main(x.view(batch_size, self.latent_size, 1, 1))
