def preprocess_model(preprocessing_steps, model):
    for step in preprocessing_steps:
        if step == 'WI':
            model.apply(weights_init)
        else:
            raise Exception('The preprocessing step \'' + step + '\' is unknown')
    return model


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    
