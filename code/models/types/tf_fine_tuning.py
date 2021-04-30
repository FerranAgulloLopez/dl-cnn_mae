import torch
import torchvision  # do not delete
import torch.nn as nn

from auxiliary_files.model_methods.model_operations import model_arq_to_json
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

        # change last layer output
        self.network.classifier[-1] = nn.Linear(self.network.classifier[-1].in_features, self.number_labels)

        # add softmax/sigmoid
        if network_config['softmax']:
            self.network.classifier.add_module('softmax', nn.Softmax(dim=1))
        else:
            self.network.classifier.add_module('sigmoid', nn.Sigmoid())

        # freeze layers if needed (only feature layers)  # TODO update to not only feature layers
        left_to_frozen = network_config['frozen_layers']
        feature_layers = list(self.network.features)
        index = 0
        while left_to_frozen > 0 and index < len(feature_layers):
            for param in list(feature_layers[index].parameters()):
                param.requires_grad = False
            left_to_frozen -= 1
            index += 1

        # reinitialize last layers if needed (only classifier layers)  # TODO update to not only classifier layers
        left_to_reinitialize = network_config['reinitialized_layers']
        classifier_layers = list(reversed(self.network.classifier))
        index = 0
        while left_to_reinitialize > 0 and index < len(classifier_layers):
            self.preprocess_net(self.config['transforms']['preprocess'], classifier_layers[index])  # reinitialize layer
            left_to_reinitialize -= 1
            index += 1
