import sys
import torch
import torch.nn as nn
from torchvision.transforms import Resize, ToPILImage, ToTensor
from torch.nn.functional import interpolate


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


class baseline_256x256_batch_norm3_maxpooling_softmax(nn.Module):
    def __init__(self, config, data_shape):
        super(baseline_256x256_batch_norm3_maxpooling_softmax, self).__init__()
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
            nn.Softmax()
        )

    def forward(self, x):
        x = self.main_cnn(x)
        return self.main_fcc(x)


class baseline_256x256_batch_norm3_maxpooling_relu(nn.Module):
    def __init__(self, config, data_shape):
        super(baseline_256x256_batch_norm3_maxpooling_relu, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.main_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 42 x 42
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 21 x 21
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 7 x 7
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 3 x 3
            nn.Flatten()
            # state size filter_scale * 16 * 3 * 3
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 16 * 3 * 3
            nn.Linear(filter_scale * 16 * 3 * 3, filter_scale * 8 * 2 * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(filter_scale * 8 * 2 * 2),
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


class baseline_256x256_batch_norm5_maxpooling_relu(nn.Module):
    def __init__(self, config, data_shape):
        super(baseline_256x256_batch_norm5_maxpooling_relu, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.main_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 42 x 42
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 21 x 21
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 7 x 7
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 16),
            # state size (filter_scale*16) x 3 x 3
            nn.Flatten()
            # state size filter_scale * 16 * 3 * 3
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 16 * 3 * 3
            nn.Linear(filter_scale * 16 * 3 * 3, filter_scale * 8 * 2 * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(filter_scale * 8 * 2 * 2),
            # state size filter_scale * 8 * 2 * 2
            nn.Linear(filter_scale * 8 * 2 * 2, filter_scale * 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(filter_scale * 4),
            # state size filter_scale * 4
            nn.Linear(filter_scale * 4, output_size),
            # state size output_size
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main_cnn(x)
        return self.main_fcc(x)


class baseline_256x256_batch_norm3_maxpooling_relu_soft(nn.Module):
    def __init__(self, config, data_shape):
        super(baseline_256x256_batch_norm3_maxpooling_relu_soft, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.main_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 42 x 42
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 21 x 21
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 7 x 7
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 3 x 3
            nn.Flatten()
            # state size filter_scale * 16 * 3 * 3
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 16 * 3 * 3
            nn.Linear(filter_scale * 16 * 3 * 3, filter_scale * 8 * 2 * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(filter_scale * 8 * 2 * 2),
            # state size filter_scale * 8 * 2 * 2
            nn.Linear(filter_scale * 8 * 2 * 2, filter_scale * 4),
            nn.ReLU(inplace=True),
            # state size filter_scale * 4
            nn.Linear(filter_scale * 4, output_size),
            # state size output_size
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.main_cnn(x)
        return self.main_fcc(x)


class baseline_256x256_batch_norm5_maxpooling(nn.Module):
    def __init__(self, config, data_shape):
        super(baseline_256x256_batch_norm5_maxpooling, self).__init__()
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
            nn.BatchNorm2d(filter_scale * 16),
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
            nn.BatchNorm1d(filter_scale * 4),
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


class Odin9(nn.Module):
    def __init__(self, config, data_shape):
        super(Odin9, self).__init__()
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
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 8 * 2 * 2),
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
        

class Odin9S(nn.Module):
    def __init__(self, config, data_shape):
        super(Odin9S, self).__init__()
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
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 8 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 8 * 2 * 2),
            # state size filter_scale * 8 * 2 * 2
            nn.Linear(filter_scale * 8 * 2 * 2, filter_scale * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size filter_scale * 4
            nn.Linear(filter_scale * 4, output_size),
            # state size output_size
            nn.Softmax()
        )

    def forward(self, x):
        x = self.main_cnn(x)
        return self.main_fcc(x)


class Odin9Dropout(nn.Module):
    def __init__(self, config, data_shape):
        super(Odin9Dropout, self).__init__()
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
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 8 * 2 * 2),
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


class Odin9Inception(nn.Module):
    def __init__(self, config, data_shape):
        super(Odin9Inception, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.inception_a = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(2, 2), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception_b = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception_c = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(9, 9), stride=(2, 2), padding=(4, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.main_cnn = nn.Sequential(
            # state size (filter_scale*3) x 64 x 64
            nn.Conv2d(filter_scale*3, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 8 * 2 * 2),
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
        inception_a = self.inception_a(x)
        inception_b = self.inception_b(x)
        inception_c = self.inception_c(x)
        x = torch.cat((inception_a, inception_b, inception_c), dim=1)
        x = self.main_cnn(x)
        return self.main_fcc(x)


class Odin9InceptionBaseline(nn.Module):
    def __init__(self, config, data_shape):
        super(Odin9InceptionBaseline, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.inception_a = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception_b = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception_c = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.main_cnn = nn.Sequential(
            # state size (filter_scale*3) x 128 x 128
            nn.Conv2d(filter_scale*3, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 64 x 64
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 32 x 32
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 16 x 16
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 32),
            # state size (filter_scale*32) x 8 x 8
            nn.Conv2d(filter_scale * 32, filter_scale * 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*64) x 4 x 4
            nn.Conv2d(filter_scale * 64, filter_scale * 128, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 128),
            nn.Flatten()
            # state size filter_scale * 128
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 128
            nn.Linear(filter_scale * 128, filter_scale * 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 64),
            # state size filter_scale * 64
            nn.Linear(filter_scale * 64, filter_scale * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size filter_scale * 32
            nn.Linear(filter_scale * 32, output_size),
            # state size output_size
            nn.Sigmoid()
        )

    def forward(self, x):
        inception_a = self.inception_a(x)
        inception_b = self.inception_b(x)
        inception_c = self.inception_c(x)
        x = torch.cat((inception_a, inception_b, inception_c), dim=1)
        x = self.main_cnn(x)
        return self.main_fcc(x)


class Odin9InceptionBaselineFc(nn.Module):
    def __init__(self, config, data_shape):
        super(Odin9InceptionBaselineFc, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.inception_a = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception_b = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception_c = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.main_cnn = nn.Sequential(
            # state size (filter_scale*3) x 128 x 128
            nn.Conv2d(filter_scale*3, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 64 x 64
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 32 x 32
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 16 x 16
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 32),
            # state size (filter_scale*32) x 8 x 8
            nn.Conv2d(filter_scale * 32, filter_scale * 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*64) x 4 x 4
            nn.Conv2d(filter_scale * 64, filter_scale * 128, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 128),
            nn.Flatten()
            # state size filter_scale * 128
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 128
            nn.Linear(filter_scale * 128, filter_scale * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 32),
            # state size filter_scale * 32
            nn.Linear(filter_scale * 32, output_size),
            # state size output_size
            nn.Sigmoid()
        )

    def forward(self, x):
        inception_a = self.inception_a(x)
        inception_b = self.inception_b(x)
        inception_c = self.inception_c(x)
        x = torch.cat((inception_a, inception_b, inception_c), dim=1)
        x = self.main_cnn(x)
        return self.main_fcc(x)


class Odin9InceptionBaselinePlus(nn.Module):
    def __init__(self, config, data_shape):
        super(Odin9InceptionBaselinePlus, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.inception_first_a = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception_first_b = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception_first_c = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.first_cnn = nn.Sequential(
            # state size (filter_scale*3) x 128 x 128
            nn.Conv2d(filter_scale*3, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 4),
            # state size (filter_scale*4) x 64 x 64
            nn.Conv2d(filter_scale * 4, filter_scale * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception_second_a = nn.Sequential(
            # state size (filter_scale*8) x 32 x 32
            nn.Conv2d(filter_scale*8, filter_scale*16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception_second_b = nn.Sequential(
            # state size (filter_scale*8) x 32 x 32
            nn.Conv2d(filter_scale*8, filter_scale*16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception_second_c = nn.Sequential(
            # state size (filter_scale*8) x 32 x 32
            nn.Conv2d(filter_scale*8, filter_scale*16, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.second_cnn = nn.Sequential(
            # state size (filter_scale*48) x 16 x 16
            nn.Conv2d(filter_scale * 48, filter_scale * 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 64),
            # state size (filter_scale*64) x 8 x 8
            nn.Conv2d(filter_scale * 64, filter_scale * 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*64) x 4 x 4
            nn.Conv2d(filter_scale * 128, filter_scale * 256, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.BatchNorm1d(filter_scale * 256)
            # state size filter_scale * 256
        )
        self.main_fcc = nn.Sequential(
            # input is filter_scale * 256
            nn.Linear(filter_scale * 256, filter_scale * 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 128),
            # state size filter_scale * 128
            nn.Linear(filter_scale * 128, filter_scale * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size filter_scale * 64
            nn.Linear(filter_scale * 64, filter_scale * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 32),
            # state size filter_scale * 32
            nn.Linear(filter_scale * 32, output_size),
            # state size output_size
            nn.Sigmoid()
        )

    def forward(self, x):
        inception_a = self.inception_first_a(x)
        inception_b = self.inception_first_b(x)
        inception_c = self.inception_first_c(x)
        x = torch.cat((inception_a, inception_b, inception_c), dim=1)
        x = self.first_cnn(x)
        inception_a = self.inception_second_a(x)
        inception_b = self.inception_second_b(x)
        inception_c = self.inception_second_c(x)
        x = torch.cat((inception_a, inception_b, inception_c), dim=1)
        x = self.second_cnn(x)
        return self.main_fcc(x)


class OdinK9(nn.Module):
    def __init__(self, config, data_shape):
        super(OdinK9, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.main_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 8 * 2 * 2),
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


class OdinN9(nn.Module):
    def __init__(self, config, data_shape):
        super(OdinN9, self).__init__()
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
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 16),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 8 * 2 * 2),
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


class Odin11(nn.Module):
    def __init__(self, config, data_shape):
        super(Odin11, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.main_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 8 * 2 * 2),
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


class OdInception9(nn.Module):
    def __init__(self, config, data_shape):
        super(OdInception9, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        self.inception1a = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception1b = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception1c = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.main_cnn = nn.Sequential(
            # state size (filter_scale*3) x 128 x 128
            nn.Conv2d(filter_scale*3, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 64 x 64
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 32 x 32
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 16 x 16
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 32)
        )
        # state size (filter_scale*32) x 8 x 8
        self.inception2a = nn.Sequential(
            # input is (filter_scale*32) x 8 x 8
            nn.Conv2d(filter_scale * 32, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception2b = nn.Sequential(
            # input is (filter_scale*32) x 8 x 8
            nn.Conv2d(filter_scale * 32, filter_scale * 8, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.inception2c = nn.Sequential(
            # input is (filter_scale*32) x 8 x 8
            nn.Conv2d(filter_scale * 32, filter_scale * 8, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2))
        )
        self.cnn_2 = nn.Sequential(
            # input is (filter_scale*32) x 4 x 4
            nn.BatchNorm2d(filter_scale * 32),
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4

        )
        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 64),
            # state size filter_scale * 64
            nn.Linear(filter_scale * 64, filter_scale * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size filter_scale * 32
            nn.Linear(filter_scale * 32, output_size),
            # state size output_size
            nn.Softmax()
        )

    def forward(self, x):
        inception1a = self.inception1a(x)
        inception1b = self.inception1b(x)
        inception1c = self.inception1c(x)
        x = torch.cat((inception1a, inception1b, inception1c), dim=1)
        x = self.main_cnn(x)
        inception2a = self.inception2a(x)
        inception2b = self.inception2b(x)
        inception2c = self.inception2c(x)
        x = torch.cat((inception2a, inception2b, inception2c), dim=1)
        x = self.cnn_2(x)
        return self.main_fcc(x)


class OdinP9(nn.Module):
    def __init__(self, config, data_shape):
        super(OdinP9, self).__init__()
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
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )
        self.small_cnn = nn.Sequential(
            # input is (number_bands) x 32 x 32
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 16 x 16
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 8 x 8
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size filter_scale * 4 * 4 * 4
            nn.Flatten()
        )
        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4 + filter_scale * 4 * 4 * 4, filter_scale * 8 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 8 * 2 * 2),
            # state size filter_scale * 8 * 2 * 2
            nn.Linear(filter_scale * 8 * 2 * 2, filter_scale * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size filter_scale * 4
            nn.Linear(filter_scale * 4, output_size),
            # state size output_size
            nn.Softmax(dim=1)
        )
        print(self)

    def forward(self, x):
        x_main = self.main_cnn(x)
        resized_x = interpolate(x, size=(32, 32))
        x_small = self.small_cnn(resized_x)

        final_x = torch.cat((x_main, x_small), dim=1)
        return self.main_fcc(final_x)


class Lassie9(nn.Module):
    def __init__(self, config, data_shape):
        super(Lassie9, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        softmax = config['softmax']
        self.main_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 16 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 16 * 4 * 4),
            # state size filter_scale * 16 * 4 * 4
            nn.Linear(filter_scale * 16 * 4 * 4, filter_scale * 16 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size filter_scale * 16 * 2 * 2
            nn.Linear(filter_scale * 16 * 2 * 2, output_size),
            # state size output_size
            nn.Softmax(dim=1) if softmax else nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main_cnn(x)
        return self.main_fcc(x)


class Lassie9B(nn.Module):
    def __init__(self, config, data_shape):
        super(Lassie9B, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        softmax = config['softmax']
        self.main_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 16 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 16 * 4 * 4),
            # state size filter_scale * 16 * 4 * 4
            nn.Linear(filter_scale * 16 * 4 * 4, filter_scale * 16 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 16 * 2 * 2),
            # state size filter_scale * 16 * 2 * 2
            nn.Linear(filter_scale * 16 * 2 * 2, output_size),
            # state size output_size
            nn.Softmax(dim=1) if softmax else nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main_cnn(x)
        return self.main_fcc(x)


class Lassie9D(nn.Module):
    def __init__(self, config, data_shape):
        super(Lassie9D, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        softmax = config['softmax']
        self.main_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale*2, filter_scale*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale*4, filter_scale*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 16 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            # state size filter_scale * 16 * 4 * 4
            nn.Linear(filter_scale * 16 * 4 * 4, filter_scale * 16 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 16 * 2 * 2),
            # state size filter_scale * 16 * 2 * 2
            nn.Linear(filter_scale * 16 * 2 * 2, output_size),
            # state size output_size
            nn.Softmax(dim=1) if softmax else nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main_cnn(x)
        return self.main_fcc(x)


class Lassie9Residual(nn.Module):
    def __init__(self, config, data_shape):
        super(Lassie9Residual, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        softmax = config['softmax']

        self.first_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
        )

        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(number_bands, filter_scale * 2, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(filter_scale * 2)
        )

        self.second_cnn = nn.Sequential(
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale * 2, filter_scale * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale * 4, filter_scale * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
        )

        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(filter_scale * 2, filter_scale * 8, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(filter_scale * 8)
        )

        self.third_cnn = nn.Sequential(
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 16 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 16 * 4 * 4),
            # state size filter_scale * 16 * 4 * 4
            nn.Linear(filter_scale * 16 * 4 * 4, filter_scale * 16 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size filter_scale * 16 * 2 * 2
            nn.Linear(filter_scale * 16 * 2 * 2, output_size),
            # state size output_size
            nn.Softmax(dim=1) if softmax else nn.Sigmoid()
        )

    def forward(self, x):
        out = self.first_cnn(x)
        out += self.down_sample_1(x)
        out2 = self.second_cnn(out)
        out2 += self.down_sample_2(out)
        out3 = self.third_cnn(out2)
        return self.main_fcc(out3)


class Lassie9ResidualO(nn.Module):
    def __init__(self, config, data_shape):
        super(Lassie9ResidualO, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        softmax = config['softmax']

        self.first_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
        )

        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(number_bands, filter_scale * 2, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(filter_scale * 2)
        )

        self.second_cnn = nn.Sequential(
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale * 2, filter_scale * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale * 4, filter_scale * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
        )

        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(filter_scale * 2, filter_scale * 8, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(filter_scale * 8)
        )

        self.third_cnn = nn.Sequential(
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 8 * 2 * 2),
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
        out = self.first_cnn(x)
        out += self.down_sample_1(x)
        out2 = self.second_cnn(out)
        out2 += self.down_sample_2(out)
        out3 = self.third_cnn(out2)
        return self.main_fcc(out3)


class Lassie9ResidualB(nn.Module):
    def __init__(self, config, data_shape):
        super(Lassie9ResidualB, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        softmax = config['softmax']

        self.first_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
        )

        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(number_bands, filter_scale * 2, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(filter_scale * 2)
        )

        self.second_cnn = nn.Sequential(
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale * 2, filter_scale * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale * 4, filter_scale * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
        )

        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(filter_scale * 2, filter_scale * 8, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(filter_scale * 8)
        )

        self.third_cnn = nn.Sequential(
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 16 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 16 * 4 * 4),
            # state size filter_scale * 16 * 4 * 4
            nn.Linear(filter_scale * 16 * 4 * 4, filter_scale * 16 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 16 * 2 * 2),
            # state size filter_scale * 16 * 2 * 2
            nn.Linear(filter_scale * 16 * 2 * 2, output_size),
            # state size output_size
            nn.Softmax(dim=1) if softmax else nn.Sigmoid()
        )

    def forward(self, x):
        out = self.first_cnn(x)
        out += self.down_sample_1(x)
        out2 = self.second_cnn(out)
        out2 += self.down_sample_2(out)
        out3 = self.third_cnn(out2)
        return self.main_fcc(out3)


class Lassie9ResidualA(nn.Module):
    def __init__(self, config, data_shape):
        super(Lassie9ResidualA, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        softmax = config['softmax']

        self.first_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
        )

        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(number_bands, filter_scale * 2, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(filter_scale * 2)
        )

        self.second_cnn = nn.Sequential(
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale * 2, filter_scale * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale * 4, filter_scale * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
        )

        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(filter_scale * 2, filter_scale * 8, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(filter_scale * 8)
        )

        self.third_cnn = nn.Sequential(
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale*32) x 4 x 4
            nn.AdaptiveAvgPool2d((1, 1)),
            # state size (filter_scale*32) x 1 x 1
            nn.Flatten()
            # state size filter_scale * 32
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32
            nn.Linear(filter_scale * 32, filter_scale * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 16),
            # state size filter_scale * 16
            nn.Linear(filter_scale * 16, filter_scale * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 8),
            # state size filter_scale * 8
            nn.Linear(filter_scale * 8, output_size),
            # state size output_size
            nn.Softmax(dim=1) if softmax else nn.Sigmoid()
        )

    def forward(self, x):
        out = self.first_cnn(x)
        out += self.down_sample_1(x)
        out2 = self.second_cnn(out)
        out2 += self.down_sample_2(out)
        out3 = self.third_cnn(out2)
        return self.main_fcc(out3)


class Lassie9ResidualBB(nn.Module):
    def __init__(self, config, data_shape):
        super(Lassie9ResidualBB, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        softmax = config['softmax']

        self.first_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
        )

        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(number_bands, filter_scale * 2, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(filter_scale * 2)
        )

        self.second_cnn = nn.Sequential(
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale * 2, filter_scale * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 4),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale * 4, filter_scale * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
        )

        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(filter_scale * 2, filter_scale * 8, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(filter_scale * 8)
        )

        self.third_cnn = nn.Sequential(
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 16),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 32),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 16 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 16 * 4 * 4),
            # state size filter_scale * 16 * 4 * 4
            nn.Linear(filter_scale * 16 * 4 * 4, filter_scale * 16 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 16 * 2 * 2),
            # state size filter_scale * 16 * 2 * 2
            nn.Linear(filter_scale * 16 * 2 * 2, output_size),
            # state size output_size
            nn.Softmax(dim=1) if softmax else nn.Sigmoid()
        )

    def forward(self, x):
        out = self.first_cnn(x)
        out += self.down_sample_1(x)
        out2 = self.second_cnn(out)
        out2 += self.down_sample_2(out)
        out3 = self.third_cnn(out2)
        return self.main_fcc(out3)


class Lassie9ResidualBBD(nn.Module):
    def __init__(self, config, data_shape):
        super(Lassie9ResidualBBD, self).__init__()
        number_bands = data_shape[0]
        filter_scale = config['filter_scale']
        output_size = config['output_size']
        softmax = config['softmax']

        self.first_cnn = nn.Sequential(
            # input is (number_bands) x 256 x 256
            nn.Conv2d(number_bands, filter_scale, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # state size (filter_scale) x 128 x 128
            nn.Conv2d(filter_scale, filter_scale*2, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale*2),
            # state size (filter_scale*2) x 64 x 64
        )

        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(number_bands, filter_scale * 2, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(filter_scale * 2)
        )

        self.second_cnn = nn.Sequential(
            # state size (filter_scale*2) x 64 x 64
            nn.Conv2d(filter_scale * 2, filter_scale * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 4),
            # state size (filter_scale*4) x 32 x 32
            nn.Conv2d(filter_scale * 4, filter_scale * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 8),
            # state size (filter_scale*8) x 16 x 16
        )

        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(filter_scale * 2, filter_scale * 8, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(filter_scale * 8)
        )

        self.third_cnn = nn.Sequential(
            # state size (filter_scale*8) x 16 x 16
            nn.Conv2d(filter_scale * 8, filter_scale * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(filter_scale * 16),
            # state size (filter_scale*16) x 8 x 8
            nn.Conv2d(filter_scale * 16, filter_scale * 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(filter_scale * 32),
            # state size (filter_scale*32) x 4 x 4
            nn.Flatten()
            # state size filter_scale * 32 * 4 * 4
        )

        self.main_fcc = nn.Sequential(
            # input is filter_scale * 32 * 4 * 4
            nn.Linear(filter_scale * 32 * 4 * 4, filter_scale * 16 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 16 * 4 * 4),
            # state size filter_scale * 16 * 4 * 4
            nn.Linear(filter_scale * 16 * 4 * 4, filter_scale * 16 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(filter_scale * 16 * 2 * 2),
            # state size filter_scale * 16 * 2 * 2
            nn.Linear(filter_scale * 16 * 2 * 2, output_size),
            # state size output_size
            nn.Softmax(dim=1) if softmax else nn.Sigmoid()
        )

    def forward(self, x):
        out = self.first_cnn(x)
        out += self.down_sample_1(x)
        out2 = self.second_cnn(out)
        out2 += self.down_sample_2(out)
        out3 = self.third_cnn(out2)
        return self.main_fcc(out3)
