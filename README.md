# ci_generative_models_comparison

## How to set up
Instructions of how to run in linux systems without conda:
- Install python3.6 and virtualenv
- Create environment: virtualenv -p /path/python3.6 venv_3.6
- Activate environment: source venv_3.6/bin/activate
- Install required libraries: pip3 install -r requirements.txt

Instructions of how to run in linux systems with conda:
- Install a conda environment with python3.6
- Activate environment
- Install the required libraries specified in the requirements.txt file

## How to run

- cd code
- python3 train_test_model.py --config path/configuration.json --output path/existant_output_directory

- python3 train_test_model.py ../input/configuration.json --output C:\Users\josej\Desktop\MAI\CI\ci_generative_models_comparison\output