import argparse
import os
from time import time
import sys
from pathlib import Path
import traceback

import train_test_model

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_configs', type=str, help="Directoy path where the config files are saved", required=True)
    parser.add_argument('--exp_outputs', type=str, help="Directory path to save the experiments results", required=True)
    parser.add_argument('--pc_id', default='unknown', type=str, help="Id to recognize the pc realizing the experiment", required=False)
    args = parser.parse_args()
    # TODO check args correctness
    return args

def count_number_files(directory):
    count = 0
    pathlist = Path(directory).glob('**/*.json')
    for path in pathlist:
         path_in_str = str(path)
         if path_in_str.endswith(".json"):
             count += 1
    return count
             

def main(input_path, output_path, pc_id):
    count = 0
    total_exps = count_number_files(input_path)
    total_time = time()
    print('Total experiments to run: ' + str(total_exps))
    pathlist = Path(input_path).glob('**/*.json')
    error = None
    for path in pathlist:
         config_path = str(path)
         filename = os.path.basename(os.path.normpath(config_path))
         if config_path.endswith(".json"):
             print('Running exp: ' + filename)
             t = time()
             results_path = output_path + '/' + filename.replace('.json', '')
             os.mkdir(results_path)
             f = open(results_path + '/log.txt', 'w')
             aux = sys.stdout
             sys.stdout = f
             try:
                 train_test_model.main(config_path, results_path, pc_id, False)
             except Exception as e:
                 error = e
                 print('ERROR', error)
                 print(traceback.format_exc())
             finally:
                 f.close()
                 sys.stdout = aux
                 count += 1
                 if error:
                     print('ERROR', error)
                     error = None
                 else:
                     print('Finished exp in ' + str(time() - t) + '. Remaining exps: ' + str(total_exps - count))
    print('Finished exps. Total time: ' + str(time() - total_time))


if __name__ == '__main__':
    args = parse_arguments()
    main(args.exp_configs, args.exp_outputs, args.pc_id)
