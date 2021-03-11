import torch
import torch.nn as nn
import numpy as np

from auxiliary_files.other_methods.visualize import compare_multiple_lines
from auxiliary_files.other_methods.util_functions import save_json


def select_loss_function(config, device):
    return eval(config['name'] + 'LossFunction')(config, device)


class LossFunction:
    def __init__(self, output_shape):
        self.output_shape = output_shape
        
    def visualize_total_losses_chart(self, visualize, filename, output_path, *losses):
        lines = []
        for name, loss in losses:
            lines.append((loss[:,0], name))
        compare_multiple_lines(visualize, list(range(0,len(loss))), lines, output_path + '/' + filename)
        
    def visualize_total_losses_file(self, filename, output_path, *losses):
        output = {'results': {}}
        for name, loss in losses:
            output['results'][name] = {'min_value': float(np.min(loss[:,0])), 'min_epoch': int(np.argmin(loss[:,0]))}
        print(output)
        save_json(output_path + '/' + filename, output)


class BetaVaeLossFunction(LossFunction):
    def __init__(self, config, device):
        super().__init__(3)
        self.beta = config['beta_kdl']
        reduction = config['reduction']
        recon_loss = config['recon_loss']
        if recon_loss == 'mse':
            self.recon_loss = nn.MSELoss(reduction=reduction)
        else:
            self.recon_loss = nn.BCELoss(reduction=reduction)
        
    def run(self, recon_x, x, mu, logvar, number_epoch):
        recon_loss = self.recon_loss(recon_x, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())    
        return recon_loss + kld*self.beta, torch.cat((recon_loss.detach().view(1), kld.detach().view(1)))
    
    def visualize_results(self, visualize, output_path, train_loss):
        lines = []
        lines.append((train_loss[:,1], 'recon'))
        lines.append((train_loss[:,2], 'kdl'))
        compare_multiple_lines(visualize, list(range(0,len(train_loss))), lines, output_path + '/recon_kdl_train_comparison')
        lines = []
        lines.append((train_loss[:,1], 'recon'))
        lines.append((train_loss[:,2]*self.beta, 'kdl'))
        compare_multiple_lines(visualize, list(range(0,len(train_loss))), lines, output_path + '/recon_kld_beta_train_comparison')


class GanLossFunction(LossFunction):
    def __init__(self, config, device):
        super().__init__(2)
        self.criterion = nn.BCELoss()
        
    def run(self, x, y):
        return self.criterion(x, y)
    
    def visualize_results(self, visualize, output_path, train_loss, test_loss):
        pass
