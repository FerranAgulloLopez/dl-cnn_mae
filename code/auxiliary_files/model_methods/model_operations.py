def model_arq_to_json(model) -> dict:
    dic = {}
    for key, value in model._modules.items():
        if len(value._modules) == 0:
            dic[key] = str(value)
        else:
            dic[key] = model_arq_to_json(value)
    return dic


def extract_model_layers(model) -> dict:
    layers = {}
    for name, module in model.named_modules():
        layers[name] = module
    return layers
