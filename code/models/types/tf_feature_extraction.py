import time
import torch
import torchvision  # do not delete
import numpy as np
import torch.nn as nn
from sklearn import *  # do not delete

from auxiliary_files.model_methods.nets import FeatureExtraction
from auxiliary_files.model_methods.model_operations import model_arq_to_json, extract_model_layers
from models.types.classifier import ClassifierModel


class TransferLearningFeatureExtraction(ClassifierModel):

    # Main methods

    # initialize model
    def __init__(self, config, data_shape, number_samples, output_path, device):
        self.config = config
        self.number_labels = data_shape[1]
        self.number_samples = number_samples  # [train, val, test]
        self.data_shape = data_shape[0]
        self.output_path = output_path
        self.device = device
        self.layers_to_extract_features_from = self.config['feature_extraction']['layers_to_extract_features_from']
        self.th_pos = self.config['feature_extraction']['discretization']['th_pos']
        self.th_neg = self.config['feature_extraction']['discretization']['th_neg']

        self.load_network()
        self.network.eval()
        self.config['network']['layers'] = model_arq_to_json(self.network)
        self.load_classifier()

    # preprocess model
    def prepare(self):
        self.network = self.network.float()
        self.network = self.network.to(self.device)

    # train the model
    def train(self, train_loader, val_loader):
        # train and validate classifier
        with torch.no_grad():
            # train
            init_time = time.time()
            output_train_features, self.true_train_labels, self.train_stats = self.extract_features(train_loader, self.number_samples[0])
            print('Elapsed time extracting train features:', time.time() - init_time, '; size:', output_train_features.shape)

            output_train_features = output_train_features.to('cpu').numpy()
            self.true_train_labels = self.true_train_labels.to('cpu').numpy()
            init_time = time.time()
            self.output_train_labels = self.train_classifier(output_train_features, self.true_train_labels)
            print('Elapsed time training classifier and predicting train labels:', time.time() - init_time)

            # validate
            init_time = time.time()
            output_val_features, self.true_val_labels, _ = self.extract_features(val_loader, self.number_samples[1], stats = self.train_stats)
            print('Elapsed time extracting val features:', time.time() - init_time, '; size:', output_val_features.shape)

            output_val_features = output_val_features.to('cpu').numpy()
            self.true_val_labels = self.true_val_labels.to('cpu').numpy()
            init_time = time.time()
            self.output_val_labels = self.predict_classifier(output_val_features)
            print('Elapsed time predicting val labels:', time.time() - init_time)

    # test the model
    def test(self, test_loader):
        with torch.no_grad():
            init_time = time.time()
            output_test_features, self.true_test_labels, _ = self.extract_features(test_loader, self.number_samples[2], stats = self.train_stats)
            print('Elapsed time extracting test features:', time.time() - init_time, '; size:', output_test_features.shape)

            output_test_features = output_test_features.to('cpu').numpy()
            self.true_test_labels = self.true_test_labels.to('cpu').numpy()
            init_time = time.time()
            self.output_test_labels = self.predict_classifier(output_test_features)
            print('Elapsed time predicting test labels:', time.time() - init_time)

    # save train results in disk
    def save_train_results(self, visualize, train_loader, val_loader):
        self.save_classification_results(self.output_path, 'train', self.true_train_labels, self.output_train_labels)
        self.save_classification_results(self.output_path, 'val', self.true_val_labels, self.output_val_labels)

    # save test results in disk
    def save_test_results(self, visualize, test_loader):
        self.save_classification_results(self.output_path, 'test', self.true_test_labels, self.output_test_labels)

    # save model in disk
    def save_model(self):
        pass

    # Auxiliary methods

    def load_network(self):
        network_config = self.config['network']

        # load architecture and weights
        self.network = eval('torchvision.models.' + network_config['name'])()  # select from torchvision models
        self.network.load_state_dict(torch.load(network_config['weights_path']))  # load weights

        # modify the network with modules inside it to extract features from
        self.total_features = 0
        last_layer_features = 0
        layers = extract_model_layers(self.network)
        layers_to_extract_features_from = set(self.layers_to_extract_features_from)
        self.feature_extraction_layers = []
        index = 0
        for key, module in self.network._modules.items():
            if isinstance(module, nn.Sequential):
                new_sequential = nn.Sequential()
                for layer in list(module):
                    new_sequential.add_module(str(index), layer)

                    if isinstance(layer, nn.Conv2d):
                        last_layer_features = layer.out_channels
                    elif isinstance(layer, nn.Linear):
                        last_layer_features = layer.out_features

                    if index in layers_to_extract_features_from:
                        feature_extraction_layer = FeatureExtraction()
                        new_sequential.add_module('feature_extractor_' + str(index), feature_extraction_layer)
                        self.feature_extraction_layers.append(feature_extraction_layer)
                        self.total_features += last_layer_features
                    index += 1
                self.network._modules[key] = new_sequential
            else:
                layers.append(module)
                index += 1

    def load_classifier(self):
        classifier_config = self.config['classifier']

        # create classifier
        self.classifier = eval(classifier_config['name'])(**classifier_config['params'])  # select from sklearn models

    def extract_features(self, data_loader, number_samples, stats=torch.empty((0, 0))):
        output_features = torch.zeros((number_samples, self.total_features)).to(self.device).detach().float()
        true_labels_array = torch.zeros(number_samples).to(self.device).detach().float()

        # extract features from dataset
        for index, (values, labels) in enumerate(data_loader, 0):  # iterate data loader
            values = values.to(self.device)
            self.network(values)

            # group layer outputs
            index_features = 0
            for feature_extraction_layer in self.feature_extraction_layers:
                layer_output = feature_extraction_layer.feature
                output_features[(index*data_loader.batch_size):(index*data_loader.batch_size + len(values)), index_features:(index_features + layer_output.shape[1])] = layer_output
                index_features += layer_output.shape[1]

            # save true labels to compute metrics afterwards
            true_labels_array[(index*data_loader.batch_size):(index*data_loader.batch_size + len(values))] = torch.argmax(labels.detach(), 1)

        # standardize
        if len(stats) == 0:
            stats = torch.zeros((2, output_features.shape[1])).to(self.device).detach().float()
            stats[0] = torch.mean(output_features, dim=0)
            stats[1] = torch.std(output_features, dim=0)
            stats[1][stats[1] == 0] = 1
        output_features = torch.div(output_features - stats[0], stats[1], out=torch.zeros_like(output_features))

        # discretize
        output_features[output_features > self.th_pos] = 1
        output_features[output_features < self.th_neg] = -1
        output_features[[(output_features >= self.th_neg) & (output_features <= self.th_pos)][0]] = 0

        return output_features, true_labels_array, stats

    def train_classifier(self, features, labels):
        self.classifier.fit(features, labels)
        return self.classifier.predict(features)

    def predict_classifier(self, features):
        return self.classifier.predict(features)

    def save_classification_results(self, path, name, labels, predicted_labels):
        print(name)
        print(metrics.confusion_matrix(labels, predicted_labels))
        print(metrics.classification_report(labels, predicted_labels))
        np.save(path + '/' + name + '_predicted_labels', predicted_labels)
