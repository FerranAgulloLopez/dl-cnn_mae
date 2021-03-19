import torch.optim as optim


def select_optimizer(config, model):
    name = config['name']
    learning_rate = config['learning_rate']['value']
    if name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate, betas=(config['beta_1'], config['beta_2']))
    elif name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate, betas=(config['beta_1'], config['beta_2']))
    elif name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=config['momentum'])
    else:
        raise Exception('The optimizer \'' + name + '\' is unknown')
