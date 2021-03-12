from data.data import Data
from data.types.mame import MAMe
from data.types.dummy import DummyData


class DataFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_data(config, *args) -> Data:
        name = config['name']
        if name == 'MAMe':
            data = MAMe(config, *args)
        elif name == 'dummy':
            data = DummyData(config, *args)
        else:
            raise Exception('The dataset with name ' + name + ' does not exist')
        if issubclass(type(data), Data):
            return data
        else:
            raise Exception('The dataset does not follow the interface definition')