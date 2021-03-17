import torch.nn as nn
import numpy as np

from auxiliary_files.other_methods.visualize import compare_multiple_lines
from auxiliary_files.other_methods.util_functions import save_json


def select_loss_function(config, device):
    return eval(config['name'] + 'LossFunction')(config, device)


class LossFunction:
    def __init__(self):
        super().__init__()
        
    def visualize_total_losses_chart(self, visualize, filename, output_path, *losses):
        lines = []
        for name, loss in losses:
            lines.append((loss, name))
        compare_multiple_lines(visualize, list(range(0,len(loss))), lines, output_path + '/' + filename)

    def visualize_total_accuracies_chart(self, visualize, filename, output_path, *accuracies):
        lines = []
        for name, accuracy in accuracies:
            lines.append((accuracy, name))
        compare_multiple_lines(visualize, list(range(0,len(accuracy))), lines, output_path + '/' + filename,
                               legend='Accuracy')
        
    def visualize_total_losses_file(self, filename, output_path, *losses):
        output = {'results': {}}
        for name, loss in losses:
            output['results'][name] = {'min_value': float(np.min(loss)), 'min_epoch': int(np.argmin(loss))}
        print(output)
        sys.stdout.flush()
        save_json(output_path + '/' + filename, output)

    def visualize_total_accuracies_file(self, filename, output_path, *accuracies):
        output = {'results': {}}
        for name, accuracies in accuracies:
            output['results'][name] = {'max_value': float(np.max(accuracies)), 'max_epoch': int(np.argmax(accuracies))}
        print(output)
        sys.stdout.flush()
        save_json(output_path + '/' + filename, output)


class DefaultClassifierLossFunction(LossFunction):
    def __init__(self, config, device):
        super().__init__()
        criterion = config['criterion']
        if criterion == 'binary_cross_entropy':
            self.criterion = nn.BCELoss()
        else:
            raise Exception('Loss function criterion not recognized')
        
    def run(self, output_labels, true_labels, number_epoch):
        return self.criterion(output_labels, true_labels)
    
    def visualize_results(self, visualize, output_path, train_loss):
        pass
