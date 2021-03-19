from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR


def select_scheduler(config, optimizer):
    name = config['name']
    if name == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer, 'min')
    elif name == 'MultiStepLR':
        return MultiStepLR(optimizer, milestones =config['milestones'])
    else:
        raise Exception('The scheduler \'' + name + '\' is unknown')
