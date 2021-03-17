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
    def __init__(self, config, data_shape, output_path, device):
        super().__init__(config, data_shape, output_path, device)
        self.config = config
        self.data_shape = data_shape[0]
        self.output_path = output_path
        self.device = device

        if config['pre'][0] == 0:
            # do not use a pretrained model
            self.config['network']['output_size'] = data_shape[1]
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
            self.not_train_epoch(number_epoch, val_loader, self.val_loss)
            val_time = time() - t
            if self.scheduler:
                if self.scheduler_loss == 'train':
                    self.scheduler.step(self.train_loss[number_epoch].detach())
                else:
                    self.scheduler.step(self.val_loss[number_epoch].detach())
            print(str('====> Epoch: {} \n' +
                      '                Train set loss: {:.6f}; Train set acc: {:.6f}; time: {} \n' +
                      '                Val set loss: {:.6f}; Val set acc: {:.6f}; time: {} \n')
                  .format(number_epoch, self.train_loss[number_epoch], self.train_acc[number_epoch], train_time,
                          self.val_loss[number_epoch], self.val_acc[number_epoch], val_time))
            sys.stdout.flush()

    # test the model
    def test(self, test_loader):
        self.test_loss = torch.from_numpy(np.zeros(1)).to(self.device).detach()
        if not self.loss_function:
            self.loss_function = select_loss_function(self.train_info['loss_function'], self.device)

        t = time()
        self.not_train_epoch(0, test_loader, self.test_loss)
        print('====> Test set loss: {:.6f}; time: {}'
              .format(self.test_loss[0], time() - t))
        sys.stdout.flush()

    # save train results in disk
    def save_train_results(self, visualize, train_loader, val_loader):
        if self.device != 'cpu':
            self.train_loss = self.train_loss.to('cpu')
            self.val_loss = self.val_loss.to('cpu')
        np.save(self.output_path + '/train_loss', self.train_loss)
        np.save(self.output_path + '/val_loss', self.val_loss)
        with torch.no_grad():
            self.loss_function.visualize_total_losses_chart(visualize, 'train_val_losses_chart', self.output_path,
                                                            ('train_loss', self.train_loss.numpy()),
                                                            ('val_loss', self.val_loss.numpy()))
            self.loss_function.visualize_total_accuracies_chart(visualize, 'train_val_accuracies_chart',
                                                                self.output_path,
                                                                ('train_accuracy', self.train_acc.numpy()),
                                                                ('val_accuracy', self.val_acc.numpy()))
            self.loss_function.visualize_total_losses_file('train_val_losses_file', self.output_path,
                                                           ('train_loss', self.train_loss.numpy()),
                                                           ('val_loss', self.val_loss.numpy()))
            self.loss_function.visualize_total_accuracies_file('train_val_accuracies_file',
                                                               self.output_path,
                                                               ('train_accuracy', self.train_acc.numpy()),
                                                               ('val_accuracy', self.val_acc.numpy()))
            self.loss_function.visualize_results(visualize, self.output_path, self.train_loss.numpy())

    # save test results in disk
    def save_test_results(self, visualize, test_loader):
        if self.device != 'cpu':
            self.test_loss = self.test_loss.to('cpu')
        np.save(self.output_path + '/test_loss', self.test_loss)
        with torch.no_grad():
            self.loss_function.visualize_total_losses_file('test_losses_file', self.output_path,
                                                           ('test_loss', self.test_loss.numpy()))

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
        self.train_loss = torch.from_numpy(np.zeros(self.number_epochs)).to(self.device).detach()
        self.val_loss = torch.from_numpy(np.zeros(self.number_epochs)).to(self.device).detach()
        self.train_acc = torch.from_numpy(np.zeros(self.number_epochs)).to(self.device).detach()
        self.val_acc = torch.from_numpy(np.zeros(self.number_epochs)).to(self.device).detach()
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
        outputs = None
        og_labels = None
        accs = torch.from_numpy(np.zeros((29,))).to(self.device).detach()

        for index, (train_values, train_labels) in enumerate(train_loader, 0):  # iterate data loader
            self.optimizer.zero_grad()
            output_labels = self.network(train_values)
            output_labels_copy = output_labels.detach()
            if outputs is None or og_labels is None:
                outputs = np.argmax(output_labels_copy, axis=1)
                og_labels = np.argmax(train_labels, axis=1)
            else:
                outputs = np.concatenate((outputs, np.argmax(output_labels_copy, axis=1)), axis=None)
                og_labels = np.concatenate((og_labels, np.argmax(train_labels, axis=1)), axis=None)
            loss = self.loss_function.run(output_labels, train_labels, number_epoch)
            loss.backward()
            self.optimizer.step()
            self.train_loss[number_epoch] = self.train_loss[number_epoch].add(
                loss.detach().view(1))  # update results' array

        self.train_loss[number_epoch] = self.train_loss[number_epoch].div(len(train_loader))  # update results' array
        for i in range(np.max(og_labels) + 1):
            this_label_index = og_labels[og_labels == i]
            this_label_outputs = outputs[this_label_index]
            this_label_labels = og_labels[this_label_index]
            accs[i] = (this_label_outputs == this_label_labels).sum() / len(this_label_labels)
        self.train_acc[number_epoch] = np.average(accs)

    # preform a not train epoch
    def not_train_epoch(self, number_epoch, data_loader, array):
        self.network.eval()
        outputs = None
        og_labels = None
        accs = torch.from_numpy(np.zeros((29,))).to(self.device).detach()

        with torch.no_grad():
            for index, (values, labels) in enumerate(data_loader, 0):  # iterate data loader
                output_labels = self.network(values)
                output_labels_copy = output_labels.detach()
                if outputs is None or og_labels is None:
                    outputs = np.argmax(output_labels_copy, axis=1)
                    og_labels = np.argmax(labels, axis=1)
                else:
                    outputs = np.concatenate((outputs, np.argmax(output_labels_copy, axis=1)), axis=None)
                    og_labels = np.concatenate((og_labels, np.argmax(labels, axis=1)), axis=None)
                loss = self.loss_function.run(output_labels, labels, number_epoch)
                array[number_epoch] = array[number_epoch].add(loss.detach().view(1))  # update results' array

        array[number_epoch] = array[number_epoch].div(len(data_loader))  # update results' array
        for i in range(np.max(og_labels) + 1):
            this_label_index = og_labels[og_labels == i]
            this_label_outputs = outputs[this_label_index]
            this_label_labels = og_labels[this_label_index]
            accs[i] = (this_label_outputs == this_label_labels).sum() / len(this_label_labels)
        self.val_acc[number_epoch] = np.average(accs)
