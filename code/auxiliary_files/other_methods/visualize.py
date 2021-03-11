import numpy as np
import matplotlib.pyplot as plt

from auxiliary_files.data_methods.preprocessing import normalize_image, normalize_dataset, reshape_as_image


#plt.switch_backend('agg')


def show_hist_matrix(nrows, ncols, hists, output, _range=None, bins=None, ticks_size=16, ylabels=[], xlabels=[], figsize=5):
    fig, axs = plt.subplots(nrows, ncols, figsize=(nrows*10, ncols*20))
    index = 0
    for row in range(nrows):
        for col in range(ncols):
           axs[row, col].hist(hists[index], bins=bins, range=_range)
           index += 1
    for index, label in enumerate(ylabels):
        axs[index, 0].set_ylabel(label, fontsize = 6*figsize)
    for index, label in enumerate(xlabels):
        axs[0, index].set_title(label, fontsize = 6*figsize)
    for ax in axs.flat:
        ax.label_outer()
        ax.tick_params(axis='x', labelsize=5*figsize)
        ax.tick_params(axis='y', labelsize=5*figsize)
    if output != None:
        plt.savefig(output, bbox_inches='tight')


def show_image(image, filename, normalize=True, color_bands=None):
    # pre: shape data -> (band width heigth)
    plt.figure(figsize=(20,10))
    if image.shape[0] == 1:
        cmap = 'gray'
    else:
        cmap = None
    plt.tight_layout()
    if normalize:
        normalize_image(image)
    plt.imshow(image, cmap=cmap)
    if filename != None:
        plt.imsave(filename, image)


def compare_sets_of_images(visualize, sets, labels, filename, figsize=5, xlabels=[]):
    # pre: shape sets -> (set number_image band width heigth)
    # pre: all sets must have the same shape
    nrows, ncols = sets.shape[0], sets.shape[1]
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*figsize, nrows*figsize))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    if sets.shape[2] == 1:
        cmap = 'gray'
    else:
        cmap = None
    for row in range(nrows):
        for col in range(ncols):
            image = sets[row, col]
            axs[row, col].imshow(image, cmap=cmap)
            axs[row, col].set_xticklabels([])
            axs[row, col].set_yticklabels([])
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
        axs[row, 0].set_ylabel(labels[row], fontsize = 5*figsize)
    for index, label in enumerate(xlabels):
        axs[0, index].set_title(label, fontsize = 5*figsize)
    if filename != None:
        plt.savefig(filename, bbox_inches='tight')
    if visualize:
        plt.show()


def plot_hist(visualize, array, filename, _type='bar'):
    plt.figure(figsize=(20,10))
    plt.hist(array, bins=50, facecolor='b', histtype=_type)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    if filename != None:
        plt.savefig(filename)
    if visualize:
        plt.show()


def print_value_counts(array):
    (unique, counts) = np.unique(array, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)


def show_matrix_of_images(visualize, data, filename, normalize=False, length_cols=5):
    # pre: shape data -> (number_image band width heigth)
    if normalize:
        data = normalize_dataset(data, inplace=False)
    nrows, ncols = data.shape[0]//length_cols, length_cols
    plt.figure(figsize=(1*length_cols, 1*nrows))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    if data.shape[1] == 1:
        plt.gray()
    index = 0
    for i in range(nrows):
        for j in range(ncols):
            ax = plt.subplot(nrows, ncols, i*ncols + j + 1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.matshow(reshape_as_image(data[index]))
            plt.xticks([])
            plt.yticks([])
            index += 1
    if filename != None:
        plt.savefig(filename, bbox_inches='tight')
    if visualize:
        plt.show()


def compare_multiple_lines(visualize, x_array, lines, path, legend=True, ylabel='Loss', title=None, ylim=None):
    # pre: the len of the input arrays must be equal to the number of epochs
    #      the input arrays must have two dimensions (data, label)
    fig, ax = plt.subplots()
    for line in lines:
        data, label = line
        ax.plot(x_array, data, label=label)
    if legend:
        ax.legend(loc='upper right', shadow=True)
    if title:
        plt.title(title)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel("Number epoch")
    plt.ylabel(ylabel)
    plt.savefig(path)
    if visualize:
        plt.show()


def compare_two_sets_of_images(visualize, set_1, set_2, filename, normalize=False):
    # pre: shape set -> (number_image band width heigth)
    #      the two sets must have the same shape
    sets = np.zeros((set_1.shape[0]*2, set_1.shape[1], set_1.shape[2], set_1.shape[3]))
    sets[:set_1.shape[0]] = set_1
    sets[-set_1.shape[0]:] = set_2
    show_matrix_of_images(visualize, sets, filename, normalize, length_cols = set_1.shape[0])
