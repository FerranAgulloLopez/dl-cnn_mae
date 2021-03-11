import argparse
import numpy as np
import torch

from factories.ModelFactory import ModelFactory
from auxiliary_files.other_methods.util_functions import load_json, save_json
from auxiliary_files.other_methods.visualize import show_matrix_of_images, compare_sets_of_images


def parse_arguments():
    def parse_bool(s: str):
        if s.casefold() in ['1', 'true', 'yes']:
            return True
        if s.casefold() in ['0', 'false', 'no']:
            return False
        raise ValueError()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help="Path to config file", required=True)
    parser.add_argument('--output', help="Path to output directory", required=True)
    parser.add_argument('--visualize', default=False, type=parse_bool, help="Show charts in standard output", required=False)
    args = parser.parse_args()
    
    # TODO check the first path is a file and a json
    # TODO check the second path is a directory
    # TODO check config file correctness
    
    return args


def load_model(model_path, data):
    model_config = load_json(model_path + '/meh.txt')
    model_name = model_config['model']['name']
    config = {'type': 'generative_model', 'name': model_name, 'pre': [1, model_path]}
    return ModelFactory.select_model(config, data.shape[1:], None, 'cpu'), model_config


def random_sampling(visualize, model, model_config, data_set, config, output):
    number = config['number']
    batch = torch.from_numpy(np.random.randn(number, model_config['model']['description']['latent_size'])).float()
    #plot_hist(visualize, batch[0].detach().to('cpu').numpy().flatten(), None)
    data = model.decode(batch)
    show_matrix_of_images(visualize, data.detach().to('cpu').numpy(), output + '/random_sampling', normalize=False, length_cols=10)


def do_generations(visualize, model, model_config, data_set, generations, output):
    for config in generations:
        name = config['name']
        if name == 'random_sampling':
            random_sampling(visualize, model, model_config, data_set, config, output)


def main(visualize, config_path, output):
    config = load_json(config_path)
    data_path = config['data_path']
    model_path = config['model_path']
    generations = config['generations']
    
    data = np.load(data_path)
    model, model_config = load_model(model_path, data)
    do_generations(visualize, model, model_config, data, generations, output)
    save_json(output + '/config', config)


if __name__ == '__main__':
    args = parse_arguments()
    main(args.visualize, args.config, args.output)
