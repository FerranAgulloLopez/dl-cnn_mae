import time
import torch
import torchvision  # do not delete
from sklearn import *  # do not delete

from models.types.classifier import ClassifierModel
from auxiliary_files.model_methods.model_operations import model_arq_to_json, extract_model_layers
from auxiliary_files.other_methods.util_functions import print_pretty_json


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

    # show model information at the start of the log
    def show_info(self):
        print_pretty_json(self.config['network'])
        print(self.classifier)
        if self.transformer:
            print(self.transformer)

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
        network_layers = extract_model_layers(self.network)
        self.features = self.Features()
        for layer in self.layers_to_extract_features_from:
            if layer in network_layers:
                network_layers[layer].register_forward_hook(self.features)
            else:
                raise Exception('Layer ' + str(layer) + ' not recognized')

    class Features:
        def __init__(self):
            self.data = []
            self.size = 0

        def __call__(self, module, module_in, module_out):
            if len(module_out.shape) > 2:
                # do spatial average pooling if it is a convolutional layer
                output = torch.mean(torch.mean(module_out, dim=2), dim=2).detach().clone()
            else:
                # nothing elsewhere
                output = module_out.detach().clone()

            self.size += output.shape[1]
            self.data.append(output)

        def clear(self):
            self.data = []
            self.size = 0

    def load_classifier(self):
        classifier_config = self.config['classifier']

        # create classifier
        self.classifier = eval(classifier_config['name'])(**classifier_config['params'])  # select from sklearn models

        # create transformer if needed
        if 'transformer' in classifier_config:
            self.transformer = eval(classifier_config['transformer']['name'])(**classifier_config['transformer']['params'])  # select from sklearn transformers
        else:
            self.transformer = None

    def extract_features(self, data_loader, number_samples, stats=torch.empty((0, 0))):
        output_features = None
        output_features_init = False
        true_labels_array = torch.zeros(number_samples).to(self.device).detach().float()

        # extract features from dataset
        for index, (values, labels) in enumerate(data_loader, 0):  # iterate data loader
            values = values.to(self.device)
            self.network(values)

            # group layer outputs
            if not output_features_init:
                output_features = torch.zeros((number_samples, self.features.size)).to(self.device).detach().float()
                output_features_init = True
            index_features = 0
            for layer_features in self.features.data:
                output_features[(index*data_loader.batch_size):(index*data_loader.batch_size + len(values)), index_features:(index_features + layer_features.shape[1])] = layer_features
                index_features += layer_features.shape[1]

            self.features.clear()

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
        if self.transformer:
            data = self.transformer.fit_transform(features)
        else:
            data = features
        self.classifier.fit(data, labels)
        return self.classifier.predict(data)

    def predict_classifier(self, features):
        if self.transformer:
            data = self.transformer.transform(features)
        else:
            data = features
        return self.classifier.predict(data)
