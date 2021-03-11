import torch

from auxiliary_files.other_methods.visualize import compare_two_sets_of_images


def model_arq_to_json(model):
    dic = {}
    for key, value in model._modules.items():
        if len(value._modules) == 0:
            dic[key] = str(value)
        else:
            dic[key] = model_arq_to_json(value)
    return dic


def encode_decode_batch(batch, model):
    with torch.no_grad():
        model.batch_size = batch.shape[0]
        mu, logvar = model.encode(batch)
        z = model.reparameterize(mu, logvar)
        return model.decode(z).cpu()


def encode_batch(batch, model, return_latent=True):
    with torch.no_grad():
        model.batch_size = batch.shape[0]
        mu, logvar = model.encode(batch.float())
        if not return_latent:
            return mu, logvar
        else:
            return model.reparameterize(mu, logvar).cpu()


def decode_batch(z, model):
    with torch.no_grad():
        model.batch_size = z.shape[0]
        return model.decode(z).cpu()


def visualize_output_examples(visualize, output_path, model, train_set, test_set):
    number_images = 10
    example_input_train = train_set[0:number_images]
    example_input_test = test_set[0:number_images]
    example_output_train = encode_decode_batch(example_input_train, model)
    example_output_test = encode_decode_batch(example_input_test, model)
    compare_two_sets_of_images(visualize, example_input_train[:number_images], example_output_train[:number_images], output_path + '/example_train_images')
    compare_two_sets_of_images(visualize, example_input_test[:number_images], example_output_test[:number_images], output_path + '/example_test_images')
