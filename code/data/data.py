import abc
import logging

from torch.utils.data import DataLoader

from auxiliary_files.other_methods.visualize import show_matrix_of_images


# Interface for the data storage
class Data(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, '__init__') and
                callable(subclass.__init__) and
                hasattr(subclass, 'prepare') and
                callable(subclass.prepare) and
                hasattr(subclass, 'get_train_loader') and
                callable(subclass.get_train_loader) and
                hasattr(subclass, 'get_val_loader') and
                callable(subclass.get_val_loader) and
                hasattr(subclass, 'get_test_loader') and
                callable(subclass.get_test_loader) and
                hasattr(subclass, 'get_data_shape') and
                callable(subclass.get_data_shape) or
                NotImplemented)

    # Main methods

    @abc.abstractmethod
    def __init__(self, config, device):
        pass

    @abc.abstractmethod
    def prepare(self):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_train_loader(self) -> DataLoader:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_val_loader(self) -> DataLoader:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_test_loader(self) -> DataLoader:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_data_shape(self) -> list:
        raise NotImplementedError('Method not implemented in interface class')

    def show_info(self):
        logging.info("Train size: " + str(len(self.get_train_loader())) + "\n" +
              "Val size: " + str(len(self.get_val_loader())) + "\n" +
              "Test size: " + str(len(self.get_test_loader())))
