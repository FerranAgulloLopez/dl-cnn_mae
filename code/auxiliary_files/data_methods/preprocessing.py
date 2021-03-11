import random
import numpy as np
from math import floor
from skimage.transform import rotate


def reshape_as_image(arr):
    im = np.ma.transpose(arr, [1, 2, 0])
    return im


def reshape_as_raster(arr):
    im = np.transpose(arr, [2, 0, 1])
    return im


def data_augmentation(steps, dataset):
    new_dataset = np.zeros((dataset.shape[0] + dataset.shape[0]*len(steps), dataset.shape[1], dataset.shape[2], dataset.shape[3]))
    new_dataset[0:dataset.shape[0]] = dataset
    cont = dataset.shape[0]
    for image in dataset:
        image = reshape_as_image(image)
        if 'rotate' in steps:
            random_angle = random.randint(1, 45)
            new_dataset[cont] = reshape_as_raster(rotate(image, angle=random_angle, mode = 'wrap'))
            cont += 1
        if 'fliplr' in steps:
            new_dataset[cont] = reshape_as_raster(np.fliplr(image))
            cont += 1
        if 'flipud' in steps:
            new_dataset[cont] = reshape_as_raster(np.flipud(image))
            cont += 1
    shuffle(new_dataset)
    return new_dataset


def shuffle(dataset):
    np.random.shuffle(dataset)


def split_array(split_size, array):
    total_number = len(array)
    if split_size <= 0 or split_size >= 1:
        raise Exception('The test size must be between 0 and 1, not included')
    cut_index = floor(total_number*(1 - split_size))
    set_1 = array[0:cut_index]
    set_2 = array[cut_index:total_number]
    return set_1, set_2


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def replace_outliers(dataset, variance_supported=4, inplace=True, superior_limits=None, inferior_limits=None):
    # pre: order shape -> [number_images number_bands width height]
    if inplace:
        output = dataset
    else:
        raise Exception('Not working properly')
        output = np.zeros(dataset.shape)
    if not superior_limits or not inferior_limits:
        mean_values = dataset.mean(axis=(3,2,0))
        std_values = dataset.std(axis=(3,2,0))
    for index_band in range(dataset.shape[1]):
        if not superior_limits or not inferior_limits:
            mean_value = mean_values[index_band]
            std_value = std_values[index_band]
            limit_superior = mean_value + variance_supported*std_value
            limit_inferior = mean_value - variance_supported*std_value
        if superior_limits:
            limit_superior = superior_limits[index_band]
        if inferior_limits:
            limit_inferior = inferior_limits[index_band]
        print('Replace outliers; band ' + str(index_band) + ' ; (superior - inferior, limit values): ', limit_superior, limit_inferior)
        for index_image in range(dataset.shape[0]):
            array = dataset[index_image][index_band]
            output[index_image][index_band] = np.where(array > limit_superior, limit_superior, array)
            output[index_image][index_band] = np.where(array < limit_inferior, limit_inferior, array)
    if not inplace:
        return output


def select_band(dataset, band):
    # pre: dataset shape -> [number_image number_band ...]
    return np.delete(dataset, [x for x in range(dataset.shape[1]) if x != band], 1)


def normalize_array(array):
    xmax, xmin = array.max(), array.min()
    div = xmax - xmin
    if div != 0:
        return (array - xmin)/(xmax - xmin)
    else:
        return array


def normalize_image(array, inplace=True):
    if inplace:
        for index in range(array.shape[0]):
            array[index] = normalize_array(array[index])
    else:
        output = np.zeros((array.shape))
        for index in range(array.shape[0]):
            output[index] = normalize_array(array[index])
        return output


def normalize_dataset(dataset, inplace=True):
    # pre: order shape -> [number_images number_bands width height]
    if inplace:
        output = dataset
    else:
        output = np.zeros(dataset.shape)
    max_values = dataset.max(axis=(3, 2, 0))
    min_values = dataset.min(axis=(3, 2, 0))
    for index_band in range(dataset.shape[1]):
        max_value = max_values[index_band]
        min_value = min_values[index_band]
        if max_value != min_value:
            for index_image in range(dataset.shape[0]):
                output[index_image][index_band] = (dataset[index_image][index_band] - min_value) / (
                            max_value - min_value)
    if not inplace:
        return output
