from models.types.classifier import ClassifierModel
from models.model import Model


class ModelFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_model(config, *args) -> Model:
        name = config['name']
        if name == 'default_classifier':
            model = ClassifierModel(config, *args)
        else:
            raise Exception('The model with name ' + name + ' does not exist')
        if issubclass(type(model), Model):
            return model
        else:
            raise Exception('The model does not follow the interface definition')