import torch.nn as nn


def model_arq_to_json(model):
    dic = {}
    for key, value in model._modules.items():
        if len(value._modules) == 0:
            dic[key] = str(value)
        else:
            dic[key] = model_arq_to_json(value)
    return dic


def extract_model_layers(model):
    layers = []
    for key, module in model._modules.items():
        if isinstance(module, nn.Sequential):
            layers += list(module)
        else:
            layers.append(module)
    return layers