import matplotlib.pyplot as plt


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
