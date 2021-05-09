import torch
import torchvision  # do not delete
import torch.nn as nn

from auxiliary_files.model_methods.model_operations import model_arq_to_json, extract_model_layers
from models.types.classifier import ClassifierModel


class TransferLearningFineTuning(ClassifierModel):

    # Main methods

    # initialize model
    def __init__(self, config, data_shape, number_samples, output_path, device):
        self.config = config
        self.number_labels = data_shape[1]
        self.number_samples = number_samples  # [train, val, test]
        self.data_shape = data_shape[0]
        self.output_path = output_path
        self.device = device

        self.load_network()
        self.config['network']['layers'] = model_arq_to_json(self.network)

    # preprocess model
    def prepare(self):
        self.network = self.network.float()
        self.network = self.network.to(self.device)

    # Auxiliary methods

    def load_network(self):
        network_config = self.config['network']

        # load architecture and weights
        self.network = eval('torchvision.models.' + network_config['name'])()  # select from torchvision models
        self.network.load_state_dict(torch.load(network_config['weights_path']))  # load weights
        layers = [layer_module for layer_name, layer_module in extract_model_layers(self.network).items() if len(list(layer_module.named_modules())) == 1]  # extract all layers in a list

        # change last layer output
        self.change_last_layer_output(network_config['name'], self.network)

        # freeze initial layers if needed
        left_to_frozen = network_config['frozen_layers']
        index = 0
        while left_to_frozen > 0 and index < len(layers):
            for param in list(layers[index].parameters()):
                param.requires_grad = False
            left_to_frozen -= 1
            index += 1

        # reinitialize last layers if needed
        left_to_reinitialize = network_config['reinitialized_layers']
        layers = list(reversed(layers))
        index = 0
        while left_to_reinitialize > 0 and index < len(layers):
            self.preprocess_net(self.config['transforms']['preprocess'], layers[index])
            left_to_reinitialize -= 1
            index += 1

    def change_last_layer_output(self, model_name, model):
        if 'resnet' in model_name:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.number_labels)
        elif 'alexnet' in model_name:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, self.number_labels)
        elif 'vgg' in model_name:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, self.number_labels)
        elif 'squeezenet' in model_name:
            model.classifier[1] = nn.Conv2d(512, self.number_labels, kernel_size=(1, 1), stride=(1, 1))
            model.num_classes = self.number_labels
        elif 'densenet' in model_name:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, self.number_labels)
        elif 'inception' in model_name:
            # handle the auxilary net
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, self.number_labels)
            # handle the primary net
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.number_labels)
        else:
            raise Exception('Unknown behaviour for changing the last layer output in this model')
