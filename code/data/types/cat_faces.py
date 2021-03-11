import os
import numpy as np
from PIL import Image

from data.fully_in_memory import FullyInMemoryData
from auxiliary_files.data_methods.preprocessing import reshape_as_raster


class CatFaces(FullyInMemoryData):

    # Main methods

    def __init__(self, config, device):
        super().__init__(config, device)
        self.directory_path = '../input/dataset/final_dataset/'
        self.resize_shape = config['resize_shape'] # width x height
        aux = config['resize_method']
        if aux == 'nearest':
            self.resize_method = Image.NEAREST
        elif aux == 'bilinear':
            self.resize_method = Image.BILINEAR
        elif aux == 'bicubic':
            self.resize_method = Image.BICUBIC
        elif aux == 'lanczos':
            self.resize_method = Image.LANCZOS
        else:
            raise Exception('The resize mehtod', aux, 'does not exist')

    def load_values(self) -> np.ndarray:
        number_images = len([name for name in os.listdir(self.directory_path) if os.path.isfile(os.path.join(self.directory_path, name))])
        print('Total images:', number_images)
        values = np.zeros((number_images, 3, self.resize_shape[0], self.resize_shape[1]))
        index = 0
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".jpg"):
                full_path = os.path.join(self.directory_path, filename)
                image = np.asarray(Image.open(full_path).resize(size=(self.resize_shape[0], self.resize_shape[1]), resample=self.resize_method))
                if len(image.shape) == 2:
                    image = np.repeat(image[:, :, np.newaxis], repeats=3, axis=2)
                values[index] = reshape_as_raster(image)
                index += 1
        return values[:100]
