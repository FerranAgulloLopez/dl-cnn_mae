from functools import partial


def preprocess_model(preprocessing_steps, model):
    for step in preprocessing_steps:
        name = step['name']
        if name == 'WI':
            model.apply(partial(weights_init, config = step))
        else:
            raise Exception('The preprocessing step \'' + step + '\' is unknown')


def weights_init(model, config):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('BatchNorm') != -1:
        _type = config['type']
        if _type == 'normal':
            model.weight.data.normal_(config['mean'], config['std'])
        elif _type == 'xavier_normal':
            model.weight.data.xavier_normal_(config['gain'])
        elif _type == 'kaiming_normal_':
            model.weight.data.kaiming_normal_(config['a'], config['mode'], config['nonlinearity'])
        if classname.find('BatchNorm') != -1:
            model.bias.data.fill_(0)
