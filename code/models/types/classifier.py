from time import time
import sys

import numpy as np
import torch

from auxiliary_files.model_methods.loss_functions import select_loss_function
from auxiliary_files.model_methods.model_operations import model_arq_to_json
from auxiliary_files.model_methods.nets import select_net
from auxiliary_files.model_methods.optimizers import select_optimizer
from auxiliary_files.model_methods.schedulers import select_scheduler
from auxiliary_files.other_methods.util_functions import print_pretty_json
from models.model import Model


class ClassifierModel(Model):

    # Main methods

    # initialize model
    def __init__(self, config, data_shape, number_samples, output_path, device):
        super().__init__(config, data_shape, number_samples, output_path, device)
        self.config = config
        self.number_labels = data_shape[1]
        self.number_samples = number_samples  # [train, val, test]
        self.data_shape = data_shape[0]
        self.output_path = output_path
        self.device = device

        if config['pre'][0] == 0:
            # do not use a pretrained model
            self.config['network']['output_size'] = self.number_labels
            self.network = select_net(self.config['network'], self.data_shape)
        else:
            # use a pretrained model
            raise Exception('Not implemented')

        self.config['network']['layers'] = model_arq_to_json(self.network)

    # preprocess model
    def prepare(self):
        self.network = self.network.float()
        self.preprocess_net(self.config['transforms']['preprocess'], self.network)
        self.network = self.network.to(self.device)

    # show model information at the start of the log
    def show_info(self):
        print_pretty_json(self.config['network'])

    # train the model
    def train(self, train_loader, val_loader):
        self.prepare_training(train_loader)

        for number_epoch in range(self.number_epochs):
            t = time()
            self.train_epoch(number_epoch, train_loader)
            train_time = time() - t
            t = time()
            self.not_train_epoch(number_epoch, val_loader, self.val_loss, self.output_val_labels, self.true_val_labels)
            val_time = time() - t
            if self.scheduler:
                if self.scheduler_loss == 'train':
                    self.scheduler.step(self.train_loss[number_epoch].detach())
                else:
                    self.scheduler.step(self.val_loss[number_epoch].detach())
            print(str('====> Epoch: {} \n' +
                      '                Train set loss: {:.6f}; Train set acc: {:.6f}; time: {} \n' +
                      '                Val set loss: {:.6f}; Val set acc: {:.6f}; time: {} \n')
                  .format(number_epoch, self.train_loss[number_epoch][0], self.train_loss[number_epoch][1], train_time,
                          self.val_loss[number_epoch][0], self.val_loss[number_epoch][1], val_time))
            sys.stdout.flush()

    # test the model
    def test(self, test_loader):
        self.test_loss = torch.from_numpy(np.zeros((1, 2))).to(self.device).detach()
        self.output_test_labels = torch.from_numpy(np.zeros((self.number_samples[2], self.number_labels))).to(self.device).detach()
        self.true_test_labels = torch.from_numpy(np.zeros((self.number_samples[2], self.number_labels))).to(self.device).detach()
        if not self.loss_function:
            self.loss_function = select_loss_function(self.train_info['loss_function'], self.device)

        t = time()
        self.not_train_epoch(0, test_loader, self.test_loss, self.output_test_labels, self.true_test_labels)
        print('====> Test set loss: {:.6f}; Accuracy: {:.6f}; time: {}'.format(self.test_loss[0][0], self.test_loss[0][1], time() - t))
        sys.stdout.flush()

    # save train results in disk
    def save_train_results(self, visualize, train_loader, val_loader):
        if self.device != 'cpu':
            self.train_loss = self.train_loss.to('cpu')
            self.val_loss = self.val_loss.to('cpu')
        np.save(self.output_path + '/train_loss', self.train_loss)
        np.save(self.output_path + '/val_loss', self.val_loss)
        with torch.no_grad():
            self.loss_function.visualize_total_losses_chart(visualize, 'train_val_losses_chart', self.output_path, 'Loss',
                                                            ('train_loss', self.train_loss[:,0].numpy()),
                                                            ('val_loss', self.val_loss[:,0].numpy()))
            self.loss_function.visualize_total_losses_chart(visualize, 'train_val_accuracies_chart',
                                                                self.output_path, 'Accuracy',
                                                                ('train_accuracy', self.train_loss[:,1].numpy()),
                                                                ('val_accuracy', self.val_loss[:,1].numpy()))
            self.loss_function.visualize_total_losses_file('train_val_losses_file', self.output_path,
                                                           ('train_loss', self.train_loss[:,0].numpy(), True),
                                                           ('val_loss', self.val_loss[:,0].numpy(), True),
                                                           ('train_accuracy', self.train_loss[:,1].numpy(), False),
                                                           ('val_accuracy', self.val_loss[:,1].numpy(), False))

    # save test results in disk
    def save_test_results(self, visualize, test_loader):
        if self.device != 'cpu':
            self.test_loss = self.test_loss.to('cpu')
        np.save(self.output_path + '/test_loss', self.test_loss)
        with torch.no_grad():
            self.loss_function.visualize_total_losses_file('test_losses_file', self.output_path,
                                                           ('test_loss', self.test_loss[:,0].numpy(), True),
                                                           ('test_accuracy', self.test_loss[:,1].numpy(), False))

    # save model in disk
    def save_model(self):
        if self.device != 'cpu':
            self.network = self.network.to('cpu')

        torch.save(self.network.state_dict(), self.output_path + '/vae_model.p')

        if self.device != 'cpu':
            self.network = self.network.to(self.device)

    # Auxiliary methods

    # prepare all the necessary objects for training the model
    def prepare_training(self, train_loader):
        self.train_info = self.config['train_info']
        self.number_epochs = self.train_info['number_epochs']
        self.loss_function = select_loss_function(self.train_info['loss_function'], self.device)
        self.train_loss = torch.from_numpy(np.zeros((self.number_epochs, 2))).to(self.device).detach()
        self.val_loss = torch.from_numpy(np.zeros((self.number_epochs, 2))).to(self.device).detach()
        self.output_train_labels = torch.from_numpy(np.zeros((self.number_samples[0], self.number_labels))).to(self.device).detach()
        self.true_train_labels = torch.from_numpy(np.zeros((self.number_samples[0], self.number_labels))).to(self.device).detach()
        self.output_val_labels = torch.from_numpy(np.zeros((self.number_samples[1], self.number_labels))).to(self.device).detach()
        self.true_val_labels = torch.from_numpy(np.zeros((self.number_samples[1], self.number_labels))).to(self.device).detach()
        self.optimizer = select_optimizer(self.train_info['optimizer'], self.network)
        if self.train_info['optimizer']['learning_rate']['dynamic'][0] == 1:
            self.scheduler = select_scheduler(self.train_info['optimizer']['learning_rate']['dynamic'][1],
                                              self.optimizer)
            if self.train_info['optimizer']['learning_rate']['dynamic'][2] == 'train':
                self.scheduler_loss = 'train'
            else:
                self.scheduler_loss = 'val'
        else:
            self.scheduler = None

    # perform a train epoch
    def train_epoch(self, number_epoch, train_loader):
        self.network.train()

        for index, (train_values, train_labels) in enumerate(train_loader, 0):  # iterate data loader
            train_values = train_values.to(self.device)
            train_labels = train_labels.to(self.device)

            self.optimizer.zero_grad()
            output_labels = self.network(train_values)
            loss = self.loss_function.run(output_labels, train_labels, number_epoch)
            loss.backward()
            self.optimizer.step()

            self.train_loss[number_epoch][0] = self.train_loss[number_epoch][0].add(loss.detach().view(1))  # update loss array
            self.output_train_labels[(index*train_loader.batch_size):(index*train_loader.batch_size + len(
                train_values))] = output_labels.detach()  # save output labels of all the epoch (to compute accuracy)
            self.true_train_labels[(index*train_loader.batch_size):(index*train_loader.batch_size + len(
                train_values))] = train_labels.detach()  # save true labels of all the epoch (to compute accuracy)

        self.train_loss[number_epoch][0] = self.train_loss[number_epoch][0].div(len(train_loader))  # update loss array
        compressed_output_train_labels = torch.argmax(self.output_train_labels, 1)
        compressed_true_train_labels = torch.argmax(self.true_train_labels, 1)
        self.train_loss[number_epoch][1] = self.compute_accuracy(compressed_output_train_labels, compressed_true_train_labels)  # update accuracy array

    # preform a not train epoch
    def not_train_epoch(self, number_epoch, data_loader, losses_array, output_labels_array, true_labels_array):
        self.network.eval()

        with torch.no_grad():
            for index, (values, labels) in enumerate(data_loader, 0):  # iterate data loader
                values = values.to(self.device)
                labels = labels.to(self.device)

                output_labels = self.network(values)
                loss = self.loss_function.run(output_labels, labels, number_epoch)

                losses_array[number_epoch][0] = losses_array[number_epoch][0].add(loss.detach().view(1))  # update loss array
                output_labels_array[(index*data_loader.batch_size):(index*data_loader.batch_size + len(values))] = output_labels.detach()  # save output labels of all the epoch (to compute accuracy)
                true_labels_array[(index*data_loader.batch_size):(index*data_loader.batch_size + len(values))] = labels.detach()  # save true labels of all the epoch (to compute accuracy)

        losses_array[number_epoch][0] = losses_array[number_epoch][0].div(len(data_loader))  # update loss array
        compressed_output_labels = torch.argmax(output_labels_array, 1)
        compressed_true_labels = torch.argmax(true_labels_array, 1)
        losses_array[number_epoch][1] = self.compute_accuracy(compressed_output_labels, compressed_true_labels)  # update accuracy array

    def compute_accuracy(self, output_labels: torch.Tensor, true_labels: torch.Tensor):
        return torch.sum(output_labels == true_labels).item()/len(output_labels)
