import sys
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
        sys.stdout.flush()
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


class baseline_256x256_maxpooling(nn.Module):
    def __init__(self, config, data_shape):
        super(baseline_256x256_maxpooling, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.main_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((3, 3)),
            # state size (filter_scale*2) x 42 x 42
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 21 x 21
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((3, 3)),
            # state size (filter_scale*8) x 7 x 7
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 3 x 3
            nn.Flatten()
            # state size filter_scale * 16 * 3 * 3
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 16 * 3 * 3
            nn.Linear(filter_scale * 16 * 3 * 3, filter_scale * 8 * 2 * 2),
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


class baseline_256x256_batch_norm3_maxpooling(nn.Module):
    def __init__(self, config, data_shape):
        super(baseline_256x256_batch_norm3_maxpooling, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.main_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((3, 3)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 42 x 42
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 21 x 21
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((3, 3)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 7 x 7
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 3 x 3
            nn.Flatten()
            # state size filter_scale * 16 * 3 * 3
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 16 * 3 * 3
            nn.Linear(filter_scale * 16 * 3 * 3, filter_scale * 8 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 8 * 2 * 2),
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


class baseline_256x256_relu(nn.Module):
    def __init__(self, config, data_shape):
        super(baseline_256x256_relu, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.main_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.ReLU(inplace=True),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3,3), stride=(3,3), padding=(1,1)),
            nn.ReLU(inplace=True),
            # state size (filter_scale*2) x 43 x 43
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            # state size (filter_scale*4) x 22 x 22
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            # state size (filter_scale*8) x 8 x 8
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            # state size (filter_scale*16) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 16 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 16 * 4 * 4
            nn.Linear(filter_scale * 16 * 4 * 4, filter_scale * 8 * 2 * 2),
            nn.ReLU(inplace=True),
            # state size filter_scale * 8 * 2 * 2
            nn.Linear(filter_scale * 8 * 2 * 2, filter_scale * 4),
            nn.ReLU(inplace=True),
            # state size filter_scale * 4
            nn.Linear(filter_scale * 4, output_size),
            # state size output_size
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main_cnn(x)
        return self.main_fcc(x)


class baseline_256x256_batch_normalization_3(nn.Module):
    def __init__(self, config, data_shape):
        super(baseline_256x256_batch_normalization_3, self).__init__()
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
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 43 x 43
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale*4) x 22 x 22
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(filter_scale * 8),
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
            nn.BatchNorm1d(filter_scale * 8 * 2 * 2),
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


class baseline_256x256_batch_normalization_5(nn.Module):
    def __init__(self, config, data_shape):
        super(baseline_256x256_batch_normalization_5, self).__init__()
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
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 43 x 43
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale*4) x 22 x 22
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 8 x 8
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(filter_scale * 16),
            # state size (filter_scale*16) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 16 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 16 * 4 * 4
            nn.Linear(filter_scale * 16 * 4 * 4, filter_scale * 8 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 8 * 2 * 2),
            # state size filter_scale * 8 * 2 * 2
            nn.Linear(filter_scale * 8 * 2 * 2, filter_scale * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 4),
            # state size filter_scale * 4
            nn.Linear(filter_scale * 4, output_size),
            # state size output_size
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main_cnn(x)
        return self.main_fcc(x)


class baseline_256x256_batch_norm3_lessdense(nn.Module):
    def __init__(self, config, data_shape):
        super(baseline_256x256_batch_norm3_lessdense, self).__init__()
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
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 43 x 43
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale*4) x 22 x 22
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 8 x 8
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (filter_scale*16) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 16 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 16 * 4 * 4
            nn.Linear(filter_scale * 16 * 4 * 4, filter_scale * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 4),
            # state size filter_scale * 4
            nn.Linear(filter_scale * 4, output_size),
            # state size output_size
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main_cnn(x)
        return self.main_fcc(x)