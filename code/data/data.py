import abc

from auxiliary_files.other_methods.visualize import show_matrix_of_images


# Interface for the data loader
class DataLoader(metaclass=abc.ABCMeta):
    
    @classmethod
    def __subclasshook__(cls, subclass): # to check that the subclasses follow the interface
        return (hasattr(subclass, 'prepare') and
                callable(subclass.prepare) and 
                hasattr(subclass, 'next') and 
                callable(subclass.next) and 
                hasattr(subclass, 'get_index') and 
                callable(subclass.get_index) and 
                hasattr(subclass, 'reset') and 
                callable(subclass.reset) and 
                hasattr(subclass, 'get_shape') and 
                callable(subclass.get_shape) and 
                hasattr(subclass, 'get_examples') and 
                callable(subclass.get_examples) or 
                NotImplemented)
        
    # Main methods
    
    @abc.abstractmethod
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def next(self):
        raise NotImplementedError('Method not implemented in interface class')
        
    @abc.abstractmethod
    def get_index(self):
        raise NotImplementedError('Method not implemented in interface class')
        
    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError('Method not implemented in interface class')
    
    @abc.abstractmethod
    def get_shape(self):
        raise NotImplementedError('Method not implemented in interface class')
        
    @abc.abstractmethod
    def get_examples(self, number):
        raise NotImplementedError('Method not implemented in interface class')


# Interface for the data storage
class Data(metaclass=abc.ABCMeta):
    
    @classmethod
    def __subclasshook__(cls, subclass): # to check that the subclasses follow the interface
        return (hasattr(subclass, '__init__') and 
                callable(subclass.__init__) and 
                hasattr(subclass, 'prepare') and 
                callable(subclass.prepare) and 
                hasattr(subclass, 'get_train_loader') and 
                callable(subclass.get_train_loader) and 
                hasattr(subclass, 'get_val_loader') and 
                callable(subclass.get_val_loader) and 
                hasattr(subclass, 'get_test_loader') and 
                callable(subclass.get_test_loader) or 
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

    def show_info(self):
        print  ("Train size: " + str(self.get_train_loader().get_shape()) + "\n" +
               "Val size: " + str(self.get_val_loader().get_shape()) + "\n" +
               "Test size: " + str(self.get_test_loader().get_shape()))
        
    def get_data_shape(self):
        return self.get_train_loader().get_shape()[1:]
    
    def show_examples(self, visualize, number, output_path):
        examples = self.get_train_loader().get_examples(number)
        try:
            m = examples.detach().to('cpu').numpy()
        except:
            m = examples
        show_matrix_of_images(visualize,m , output_path, normalize=True)
