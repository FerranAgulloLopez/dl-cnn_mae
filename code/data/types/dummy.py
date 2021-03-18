import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from data.data import Data


class DummyData(Data):
    def __init__(self, config, device):
        super().__init__(config, device)
        values = np.zeros((100, 3, 256, 256))
        labels = np.zeros((100, 20))
        tensor_x = torch.from_numpy(values).float()
        tensor_y = torch.from_numpy(labels).float()
        self.data_loader = TensorDataset(tensor_x, tensor_y)
        self.batch_size = 64

    def get_train_loader(self) -> DataLoader:
        return DataLoader(dataset=self.data_loader, shuffle=True, batch_size=self.batch_size)

    def get_test_loader(self) -> DataLoader:
        return DataLoader(dataset=self.data_loader, shuffle=True, batch_size=self.batch_size)

    def get_val_loader(self) -> DataLoader:
        return DataLoader(dataset=self.data_loader, shuffle=True, batch_size=self.batch_size)

    def prepare(self):
        pass

    def get_data_shape(self):
        return [[3, 256, 256], 20]  # image size, number_labels

    def get_number_samples(self):
        return [100, 100, 100]  # train, val, test
