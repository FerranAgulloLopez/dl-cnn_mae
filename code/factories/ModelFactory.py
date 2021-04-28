from models.model import Model
from models.types.classifier import ClassifierModel
from models.types.tf_fine_tuning import TransferLearningFineTuning
from models.types.tf_feature_extraction import TransferLearningFeatureExtraction


class ModelFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_model(config, *args) -> Model:
        name = config['name']
        if name == 'default_classifier':
            model = ClassifierModel(config, *args)
        elif name == 'tf_fine_tuning':
            model = TransferLearningFineTuning(config, *args)
        elif name == 'tf_feature_extraction':
            model = TransferLearningFeatureExtraction(config, *args)
        else:
            raise Exception('The model with name ' + name + ' does not exist')
        if issubclass(type(model), Model):
            return model
        else:
            raise Exception('The model does not follow the interface definition')
