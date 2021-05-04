import torch.nn as nn


def model_arq_to_json(model):
    dic = {}
    for key, value in model._modules.items():
        if len(value._modules) == 0:
            dic[key] = str(value)
        else:
            dic[key] = model_arq_to_json(value)
    return dic


def recursive_extract_iterable_items(_object):
    try:
        iterable = iter(_object)
        output = []
        for item in list(_object):
            if isinstance(item, nn.Module):
                output += recursive_extract_iterable_items(item._modules.items())
            else:
                output += recursive_extract_iterable_items(item[1])
        return output
    except TypeError as _:
        return [_object]


def extract_model_layers(model):
    return recursive_extract_iterable_items(model._modules.items())
