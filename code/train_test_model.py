import torch
import argparse
import sys
from time import time
import random
import torch.backends.cudnn as cudnn
import numpy as np
from auxiliary_files.other_methods.util_functions import load_json, save_json
from factories.ModelFactory import ModelFactory
from factories.DataFactory import DataFactory

def parse_arguments():
    def parse_bool(s: str):
        if s.casefold() in ['1', 'true', 'yes']:
            return True
        if s.casefold() in ['0', 'false', 'no']:
            return False
        raise ValueError()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, help="Path to config file", required=True)
    parser.add_argument('--output', type=str, help="Path to output directory", required=True)
    parser.add_argument('--redirect', default=False, type=parse_bool, help="Redirect output to output directory", required=False)
    args = parser.parse_args()
    
    # TODO check the first path is a file and a json
    # TODO check the second path is a directory
    # TODO check config file correctness
    
    return args


def main(config_file: str, output_path: str, visualize: bool):
    # visualize -> show output charts in standard output

    # Load configuration file
    config = load_json(config_file)
    save_json(output_path + '/initial_config', config)
    
    # Config setup
    if int(config['manual_seed']) == -1:
        config['manual_seed'] = random.randint(1, 10000)
    random.seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])
    torch.manual_seed(config['manual_seed'])
    cudnn.benchmark = True
    
    # Load data
    print('Loading data')
    data = DataFactory.select_data(config['data'], config['device'])
    data.prepare()
    data.show_info()
    # data.show_examples(visualize, 10, output_path + '/data_examples')
    train_loader = data.get_train_loader()
    val_loader = data.get_val_loader()
    test_loader = data.get_test_loader()

    # Load model
    print('\nLoading model')
    model = ModelFactory.select_model(config['model'], data.get_data_shape(), output_path, config['device'])
    model.show_info()
    model.prepare()
    
    # Train model
    print('\nTraining model')
    t = time()
    model.train(train_loader, val_loader)
    train_elapsed_time = time() - t
    
    # Test model
    print('\nTesting model')
    t = time()
    model.test(test_loader)
    test_elapsed_time = time() - t
    
    # Visualize results
    print('\nGenerating visualizations')
    model.save_train_results(visualize, train_loader, val_loader)
    model.save_test_results(visualize, test_loader)
    
    # Save results and model
    print('\nSaving model')
    model.save_model()
    
    # Save config
    config['train_elapsed_time'] = train_elapsed_time
    config['test_elapsed_time'] = test_elapsed_time
    save_json(output_path + '/config', config)


if __name__ == '__main__':

    # Get input params (input config and output paths)
    args = parse_arguments()
    
    # Redirect program output
    if args.redirect:
        f = open(args.output + '/log.txt', 'w')
        sys.stdout = f
    
    # Run main program
    main(args.config, args.output, not args.redirect)
    if args.redirect:
        f.close()

