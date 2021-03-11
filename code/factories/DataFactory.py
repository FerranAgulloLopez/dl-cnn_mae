from data.data import Data
from data.types.cat_faces import CatFaces

class DataFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_data(config, *args) -> Data:
        name = config['name']
        if name == 'cat_faces':
            data = CatFaces(config, *args)
        else:
            raise Exception('The dataset with name ' + name + ' does not exist')
        if issubclass(type(data), Data):
            return data
        else:
            raise Exception('The dataset does not follow the interface definition')