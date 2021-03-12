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


class baseline_256x256(nn.Module):
    def __init__(self, config, data_shape):
        super(baseline_256x256, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.main_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3,3), stride=(3,3), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale*2) x 43 x 43
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale*4) x 22 x 22
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale*8) x 8 x 8
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale*16) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 16 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 16 * 4 * 4
            nn.Linear(filter_scale * 16 * 4 * 4, filter_scale * 8 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size filter_scale * 8 * 2 * 2
            nn.Linear(filter_scale * 8 * 2 * 2, filter_scale * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size filter_scale * 4
            nn.Linear(filter_scale * 4, output_size),
            # state size output_size
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main_cnn(x)
        return self.main_fcc(x)
