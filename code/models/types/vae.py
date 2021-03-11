import torch
import torch.nn as nn
import numpy as np
from time import time
import os
import shutil

from models.generative_model import GenerativeModel
from auxiliary_files.model_methods.optimizers import select_optimizer
from auxiliary_files.model_methods.schedulers import select_scheduler
from auxiliary_files.model_methods.loss_functions import select_loss_function
from auxiliary_files.model_methods.nets import select_net
from auxiliary_files.other_methods.visualize import compare_two_sets_of_images, compare_multiple_lines, plot_hist
from auxiliary_files.other_methods.util_functions import load_json, print_pretty_json
from auxiliary_files.model_methods.model_operations import model_arq_to_json


class VAENetwork(nn.Module):
    def __init__(self, encoder_config, decoder_config, data_shape):
        super(VAENetwork, self).__init__()
        self.encoder = select_net(encoder_config, data_shape)
        self.decoder = select_net(decoder_config, data_shape)
        
    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def forward(self, x):
        batch_size = x.shape[0]
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z, batch_size), mu, logvar, z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x, x.shape[0])


class VAEGenerativeModel(GenerativeModel):
    
    # Main methods
    
    def __init__(self, config, data_shape, output_path, device):
        super().__init__(config, data_shape, output_path, device)
        self.config = config
        self.data_shape = data_shape
        self.output_path = output_path
        self.device = device
        
        if config['pre'][0] == 0: 
            # Don't use a pretrained model
            self.description = config['description']
            self.latent_size = self.description['latent_size']
            self.description['encoder']['latent_size'] = self.latent_size # Pass latent size info to encoder and decoder builders
            self.description['decoder']['latent_size'] = self.latent_size
            self.vae_model = VAENetwork(self.description['encoder'], self.description['decoder'], data_shape)
        else: 
            # Use a pretrained model
            model_directory = config['pre'][1]
            model_config = load_json(model_directory + '/meh.txt')
            self.description = model_config['model']['description'] # TODO check there is no shape problems related to the new data shape
            self.latent_size = self.description['latent_size']
            self.vae_model = VAENetwork(self.description['encoder'], self.description['decoder'], data_shape)
            self.vae_model.load_state_dict(torch.load(model_directory + '/vae_model.p'))
        
        self.description['encoder']['net'] = model_arq_to_json(self.vae_model.encoder) # Show net modules in output config file
        self.description['decoder']['net'] = model_arq_to_json(self.vae_model.decoder)
        
    def prepare(self):
        self.transforms = self.config['transforms']
        self.vae_model = self.vae_model.float()
        self.preprocess_net(self.transforms['preprocess'], self.vae_model)
        self.vae_model = self.vae_model.to(self.device)
        
    def show_info(self):
        print_pretty_json(self.description)
        
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
                    self.scheduler.step(self.train_loss[number_epoch][0].detach())
                else:
                    self.scheduler.step(self.val_loss[number_epoch][0].detach())
            print(str('====> Epoch: {} \n' +
                  '                Train set loss: {:.6f}; time: {} \n' +
                  '                Val set loss: {:.6f}; time: {}')
                  .format(number_epoch, self.train_loss[number_epoch][0], train_time, self.val_loss[number_epoch][0], val_time))
            
    def test(self, test_loader):
        self.test_loss = torch.from_numpy(np.zeros((1, self.loss_function.output_shape))).to(self.device).detach()
        if not self.loss_function:
            self.loss_function = select_loss_function(self.train_info['loss_function'], self.device)
        
        t = time()
        self.not_train_epoch(0, test_loader, self.test_loss)
        print('====> Test set loss: {:.6f}; time: {}'
              .format(self.test_loss[0][0], time() - t))
        
    def save_train_results(self, visualize, train_loader, val_loader):
        self.train_data_to_cpu()
        np.save(self.output_path + '/train_loss', self.train_loss)
        np.save(self.output_path + '/val_loss', self.val_loss)
        with torch.no_grad():
            self.loss_function.visualize_total_losses_chart(visualize, 'train_val_losses_chart', self.output_path, ('train_loss', self.train_loss.numpy()), ('val_loss', self.val_loss.numpy()))
            self.loss_function.visualize_total_losses_file('train_val_losses_file', self.output_path, ('train_loss', self.train_loss.numpy()), ('val_loss', self.val_loss.numpy()))
            self.loss_function.visualize_results(visualize, self.output_path, self.train_loss.numpy())
            self.visualize_examples(visualize, 10, train_loader, 'reconstruct_examples_train')
            self.visualize_examples(visualize, 10, val_loader, 'reconstruct_examples_val')
            self.visualize_latent(visualize, train_loader, val_loader)

    def save_test_results(self, visualize, test_loader):
        if self.device != 'cpu':
            self.test_loss = self.test_loss.to('cpu')
        np.save(self.output_path + '/test_loss', self.test_loss)
        with torch.no_grad():
            self.loss_function.visualize_total_losses_file('test_losses_file', self.output_path, ('test_loss', self.test_loss.numpy()))
            self.visualize_examples(visualize, 10, test_loader, 'reconstruct_examples_test')
            if 'latent_last' in self.config_visualize_latent and self.config_visualize_latent['latent_last'][0]:
                encoded_test_set = torch.from_numpy(np.zeros((test_loader.get_shape()[0], self.latent_size))).to(self.device).detach()
                self.encode_set(test_loader, encoded_test_set)
                np.save(self.output_path + '/encoded_test_set', encoded_test_set.to('cpu').numpy())
            
    def save_model(self):
        if self.device != 'cpu':
            self.vae_model = self.vae_model.to('cpu')

        torch.save(self.vae_model.state_dict(), self.output_path + '/vae_model.p')
        
        if self.device != 'cpu':
            self.vae_model = self.vae_model.to(self.device)
            
    def encode(self, data):
        self.vae_model.eval()
        with torch.no_grad():
            return self.vae_model.encode(data)
        
    def decode(self, data):
        self.vae_model.eval()
        with torch.no_grad():
            return self.vae_model.decode(data)
        
    def reconstruct(self, data):
        self.vae_model.eval()
        with torch.no_grad():
            output, _, _, _ = self.vae_model(data)
            return output
        
    # Auxiliary methods
    
    def prepare_training(self, train_loader):
        self.train_info = self.config['train_info']
        self.number_epochs = self.train_info['number_epochs']
        self.loss_function = select_loss_function(self.train_info['loss_function'], self.device)
        self.train_loss = torch.from_numpy(np.zeros((self.number_epochs, self.loss_function.output_shape))).to(self.device).detach()
        self.val_loss = torch.from_numpy(np.zeros((self.number_epochs, self.loss_function.output_shape))).to(self.device).detach()
        self.optimizer = select_optimizer(self.train_info['optimizer'], self.vae_model)
        if self.train_info['optimizer']['learning_rate']['dynamic'][0] == 1:
            self.scheduler = select_scheduler(self.train_info['optimizer']['learning_rate']['dynamic'][1], self.optimizer)
            if self.train_info['optimizer']['learning_rate']['dynamic'][2] == 'train':
                self.scheduler_loss = 'train'
            else:
                self.scheduler_loss = 'val'
        else:
            self.scheduler = None
        self.config_visualize_latent = self.train_info['visualize_latent']
        self.prepare_latent_data(train_loader)
        
    def save_train_data(self):
        np.save(self.output_path + '/train_loss', self.train_loss)
        np.save(self.output_path + '/val_loss', self.val_loss)
        
        # visualize latent
        if self.bool_latent_std_mean:
            np.save(self.output_path + '/z_std_mean', self.z_std_mean)
        if self.bool_latent_last:
            pass
            
    def prepare_latent_data(self, train_loader):
        self.bool_latent_std_mean = False
        if self.config_visualize_latent['latent_std_mean'] == 1:
            self.bool_latent_std_mean = True
            self.z_std_mean = torch.from_numpy(np.zeros((2, self.number_epochs, self.latent_size))).to(self.device).detach()
            
    def train_epoch(self, number_epoch, train_loader):
        self.vae_model.train()
        train_loader.reset()
        index = train_loader.get_index()
        train_data = train_loader.next()

        while train_data: # iterate data loader
            train_values, _ = train_data
            
            self.optimizer.zero_grad()
            recon_data, mu, logvar, z = self.vae_model(train_values)
            loss, other_losses = self.loss_function.run(recon_data, train_values, mu, logvar, number_epoch)
            loss.backward()
            self.optimizer.step()

            self.train_loss[number_epoch] = self.train_loss[number_epoch].add(torch.cat((loss.detach().view(1), other_losses)))
            self.update_latent_data(number_epoch, mu, logvar, z, index, train_loader.get_shape()[0]) # To do the latent charts at the end
            
            index = train_loader.get_index()
            train_data = train_loader.next()
        self.train_loss[number_epoch] = self.train_loss[number_epoch].div(train_loader.get_shape()[0])
        
    def not_train_epoch(self, number_epoch, data_loader, array):
        self.vae_model.eval()
        data_loader.reset()
        data = data_loader.next()
        
        with torch.no_grad():
            while data: # iterate data loader
                values, _ = data                
                recon_batch, mu, logvar, z = self.vae_model(values)
                loss, other_losses = self.loss_function.run(recon_batch, values, mu, logvar, number_epoch)
                array[number_epoch] = array[number_epoch].add(torch.cat((loss.detach().view(1), other_losses)))
                
                data = data_loader.next()
        
        array[number_epoch] = array[number_epoch].div(data_loader.get_shape()[0])
        
    def encode_set(self, data_loader, array):
        self.vae_model.eval()
        data_loader.reset()
        index = data_loader.get_index()
        data = data_loader.next()
        
        with torch.no_grad():
            while data: # iterate data loader
                values, _ = data                
                mu, _ = self.vae_model.encode(values)
                array[index:data_loader.get_index()] = mu
                index = data_loader.get_index()
                data = data_loader.next()
            
    def update_latent_data(self, number_epoch, mu, logvar, z, index, lenght_set):
        if self.bool_latent_std_mean:
                self.update_latent_std_mean_values(number_epoch, index, lenght_set, z.detach(), self.z_std_mean)
                
    def update_latent_std_mean_values(self, number_epoch, index, lenght_set, mu, array):
        # algorithm from wikipedia, standard deviation - rapid calculation methods
        batch_size = mu.shape[0]
        for elem in range(batch_size): # TODO do all the pacth at the same time
            index += 1
            mu_elem = mu[elem]
            aux = array[0][number_epoch].clone()
            array[0][number_epoch] += (mu_elem - aux)/index
            array[1][number_epoch] += (mu_elem - aux)*(mu_elem - array[0][number_epoch])
        last = batch_size > (lenght_set - index)
        if last:
            array[1][number_epoch] = torch.sqrt(array[1][number_epoch]/(index - 1))
    
    def train_data_to_cpu(self):
        if self.device != 'cpu':
            self.train_loss = self.train_loss.to('cpu')
            self.val_loss = self.val_loss.to('cpu')
            if self.bool_latent_std_mean:
                self.z_std_mean = self.z_std_mean.to('cpu') 
    
    def visualize_examples(self, visualize, number, data_loader, filename):
        examples_input = data_loader.get_examples(number)
        examples_output = self.reconstruct(examples_input)
        compare_two_sets_of_images(visualize, examples_input.detach().to('cpu').numpy(), examples_output.detach().to('cpu').numpy(), self.output_path + '/' + filename)
        
    def visualize_latent(self, visualize, train_loader, val_loader):
        if os.path.exists(self.output_path + '/latent'):
            shutil.rmtree(self.output_path + '/latent')
        os.mkdir(self.output_path + '/latent')
        if self.config_visualize_latent['latent_std_mean'] == 1:
            path = self.output_path + '/latent' + '/mean_std'
            os.mkdir(path)
            self.visualize_latent_mean_std(visualize, self.z_std_mean, path + '/z', 'z')
            np.save(self.output_path + '/latent' + '/mean_std' + '/z' + '/Evolution latent space: z', self.z_std_mean)
        if 'latent_last' in self.config_visualize_latent and self.config_visualize_latent['latent_last'][0]:
                encoded_train_set = torch.from_numpy(np.zeros((train_loader.get_shape()[0], self.latent_size))).to(self.device).detach()
                self.encode_set(train_loader, encoded_train_set)
                np.save(self.output_path + '/encoded_train_set', encoded_train_set.to('cpu').numpy())
                encoded_val_set = torch.from_numpy(np.zeros((val_loader.get_shape()[0], self.latent_size))).to(self.device).detach()
                self.encode_set(val_loader, encoded_val_set)
                np.save(self.output_path + '/encoded_val_set', encoded_val_set.to('cpu').numpy())
                    
    def visualize_latent_mean_std(self, visualize, array, path, name):
        os.mkdir(path)
        lines = []
        for dimension in range(array.shape[2]):
            lines.append((array[0,:,dimension].numpy(), 'dimension_' + str(dimension)))
        compare_multiple_lines(visualize, list(range(0,array.shape[1])), lines, path + '/' + name + '_mean', legend=False, ylabel='mean')
        lines = []
        for dimension in range(array.shape[2]):
            lines.append((array[1,:,dimension].numpy(), 'dimension_' + str(dimension)))
        compare_multiple_lines(visualize, list(range(0,array.shape[1])), lines, path + '/' + name + '_std', legend=False, ylabel='std')
        plot_hist(visualize, array[0, self.number_epochs-1, :].numpy(), path + '/' + name + '_mean_hist_last_epoch')
        plot_hist(visualize, array[1, self.number_epochs-1, :].numpy(), path + '/' + name + '_std_hist_last_epoch')
