import sys
import torch
import torch.nn as nn
import numpy as np

from auxiliary_files.other_methods.visualize import compare_multiple_lines
from auxiliary_files.other_methods.util_functions import save_json


def select_loss_function(config, device):
    return eval(config['name'] + 'LossFunction')(config, device)


class LossFunction:
    def __init__(self):
        super().__init__()
        
    def visualize_total_losses_chart(self, visualize, filename, output_path, ylabel, *losses):
        lines = []
        for name, loss in losses:
            lines.append((loss, name))
        compare_multiple_lines(visualize, list(range(0,len(loss))), lines, output_path + '/' + filename, ylabel=ylabel)

    def visualize_total_losses_file(self, filename, output_path, *losses):
        output = {'results': {}}
        for name, loss, min_value in losses:
            if min_value:
                output['results'][name] = {'min_value': float(np.min(loss)), 'min_epoch': int(np.argmin(loss))}
            else:
                output['results'][name] = {'max_value': float(np.max(loss)), 'max_epoch': int(np.argmax(loss))}
        print(output)
        sys.stdout.flush()
        save_json(output_path + '/' + filename, output)


class DefaultClassifierLossFunction(LossFunction):
    def __init__(self, config, device):
        super().__init__()
        self.criterion_name = config['criterion']
        criterion = config['criterion']
        if criterion == 'binary_cross_entropy':
            self.criterion = nn.BCELoss()
        elif criterion == 'negative_log_likelihood':
            self.criterion = nn.NLLLoss()
        elif criterion == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise Exception('Loss function criterion not recognized')
        
    def run(self, output_labels, true_labels, number_epoch):
        if self.criterion_name != 'binary_cross_entropy':
            true_labels = torch.argmax(true_labels, 1)
        return self.criterion(output_labels, true_labels)
