from models.types.vae import VAEGenerativeModel
from models.generative_model import GenerativeModel


class ModelFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_model(config, *args) -> GenerativeModel:
        name = config['name']
        if name == 'vae':
            model = VAEGenerativeModel(config, *args)
        else:
            raise Exception('The model with name ' + name + ' does not exist')
        if issubclass(type(model), GenerativeModel):
            return model
        else:
            raise Exception('The model does not follow the interface definition')