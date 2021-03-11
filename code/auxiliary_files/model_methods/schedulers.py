from torch.optim.lr_scheduler import ReduceLROnPlateau


def select_scheduler(config, optimizer):
    name = config['name']
    if name == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer, 'min')
    else:
        raise Exception('The scheduler \'' + name + '\' is unknown')
