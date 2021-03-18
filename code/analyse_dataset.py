import os
from PIL import Image
import numpy as np
import math

DIRECTORY_PATH = '../input/MAMe/MAMe_data_256/data_256'
OUTPUT_PATH = '../output'

max = np.asarray([0, 0, 0])
min = np.asarray([0, 0, 0])
mean = np.asarray([0, 0, 0])
histogram = [np.zeros(256), np.zeros(256), np.zeros(256)]

count = 0
for filename in os.listdir(DIRECTORY_PATH):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        print(count)
        # load image
        file_path = os.path.join(DIRECTORY_PATH, filename)
        image = np.asarray(Image.open(file_path))
        image = np.transpose(image, [2, 0, 1]) # shape -> channels height width

        # compute stuff
        max = np.maximum(max, image.max(axis=(1, 2)))
        min = np.maximum(min, image.min(axis=(1, 2)))
        #welford[0].consume(image[0].flatten())
        #welford[1].consume(image[1].flatten())
        #welford[2].consume(image[2].flatten())

        count += 1


# Show results
print('Max:', max)
print('Min:', min)
print('Mean:', [welford[0].mean, welford[1].mean, welford[2].mean])
print('Std:', [welford[0].std, welford[1].std, welford[2].std])

