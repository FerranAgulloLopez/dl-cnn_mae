import abc

from auxiliary_files.model_methods.model_preprocessing import preprocess_model


# Interface for generative models
class GenerativeModel(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'preprocess') and
                callable(subclass.preprocess) and
                hasattr(subclass, 'show_info') and
                callable(subclass.show_info) and
                hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'test') and
                callable(subclass.test) and
                hasattr(subclass, 'save_train_results') and
                callable(subclass.save_train_results) and
                hasattr(subclass, 'save_test_results') and
                callable(subclass.save_test_results) and
                hasattr(subclass, 'save_model') and
                callable(subclass.save_model) and
                hasattr(subclass, 'encode') and
                callable(subclass.encode) or
                NotImplemented)

    # Main methods

    @abc.abstractmethod
    def __init__(self, config, data_shape, output_path, device):
        pass

    @abc.abstractmethod
    def prepare(self):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def show_info(self):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def train(self, train_loader, val_loader):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def test(self, test_data):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def save_train_results(self, visualize, train_loader, val_loader):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def save_test_results(self, visualize, test_loader):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def save_model(self):
        raise NotImplementedError('Method not implemented in interface class')

    # Auxiliary methods

    def preprocess_net(self, preprocessing_steps, model):
        preprocess_model(preprocessing_steps, model)
        
    @abc.abstractmethod
    def encode(self, data):
        raise NotImplementedError('Method not implemented in interface class')
        
    @abc.abstractmethod
    def decode(self, data):
        raise NotImplementedError('Method not implemented in interface class')
        
    @abc.abstractmethod
    def reconstruct(self, data):
        raise NotImplementedError('Method not implemented in interface class')
