import torch
import torch.nn as nn

WEIGHTS_PATH = './weights.p'


class Odin9(nn.Module):
    def __init__(self):
        super(Odin9, self).__init__()
        number_bands = 3
        filter_scale = 32
        output_size = 29
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

   
if __name__ == '__main__':
    model = Odin9()
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    print('Model loaded')

