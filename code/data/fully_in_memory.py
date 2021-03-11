import abc
import torch
import numpy as np

from data.data import Data, DataLoader
from auxiliary_files.data_methods.preprocessing import split_array, shuffle, normalize_dataset, replace_outliers, data_augmentation


class FullInMemoryDataLoader(DataLoader):

    def __init__(self, data, batch_size):
        super().__init__()
        self.data = data
        self.iter = 0
        self.batch_size = batch_size if batch_size > 0 else len(data)

    def next(self):
        if self.iter >= len(self.data):
            return None
        else:
            output = self.data[self.iter:(self.iter + self.batch_size)]
            self.iter += self.batch_size
            return (output, None)

    def get_index(self):
        return self.iter

    def reset(self):
        self.iter = 0

    def get_shape(self):
        return self.data.shape

    def get_examples(self, number):
        if number > len(self.data):
            raise Exception('The demanded number of examples exceeds the data length')
        else:
            return self.data[0:number]


class FullyInMemoryData(Data):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'load_values') and
                callable(subclass.load_values) or
                NotImplemented)

    # Abstract class main methods

    def __init__(self, config, device):
        super().__init__(config, device)
        self.config = config
        self.device = device
    
    def prepare(self):
        values = self.load_values()
        values = self.preprocess(self.config['transforms']['preprocess'], values, self.device)
        train_values, val_values, test_values = self.split(self.config['split'], values)
        if 'data_augmentation' in self.config['transforms'] and self.config['transforms']['data_augmentation'][0] == 1:
            train_values = data_augmentation(self.config['transforms']['data_augmentation'][1], train_values)
        train_values, val_values, test_values = self.prepare_data(train_values, val_values, test_values, self.device)
        batch_size = self.config['batch_size']
        self.train_loader = FullInMemoryDataLoader(train_values, batch_size)
        self.val_loader = FullInMemoryDataLoader(val_values, batch_size)
        self.test_loader = FullInMemoryDataLoader(test_values, batch_size)
        
    def get_train_loader(self) -> DataLoader:
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        return self.val_loader
        
    def get_test_loader(self) -> DataLoader:
        return self.test_loader
    
    # Abstract class auxiliary methods
    
    def preprocess(self, config_preprocess, values, device):
        for step in config_preprocess:
            name = step['name']
            if name == 'normalize':
                normalize_dataset(values)
            elif name == 'shuffle':
                shuffle(values)
            elif name == 'replace_outliers':
                replace_outliers(values)
            else:
                raise Exception('The preprocessing step \'' + name + '\' is unknown')
        return values

    def prepare_data(self, train_values, val_values, test_values, device):

        def private_prepare(values):
            values = torch.from_numpy(values)
            values = values.float()
            if device != 'cpu':
                values = values.to(device)
            return values

        train_values = private_prepare(train_values)
        val_values = private_prepare(val_values)
        test_values = private_prepare(test_values)

        return train_values, val_values, test_values

    
    def split(self, config_split, values):
        train_size = config_split['train_size']
        val_size = config_split['val_size']
        test_size = config_split['test_size']
        train_values, not_train_values = split_array(1 - train_size, values)
        val_values, test_values = split_array(test_size/(val_size + test_size), not_train_values)
        return train_values, val_values, test_values

    # Subclass main methods

    @abc.abstractmethod
    def load_values(self) -> np.ndarray:
        raise NotImplementedError('Method not implemented in interface class')
