import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data.data import Data


class MAMeDataset(Dataset):
    def __init__(self, metadata_path, key, root_dir, split, transform):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['Subset'] == self.split].reset_index(drop=True)
        self.metadata = self.metadata[['Image file', 'Medium']]
        self.metadata['Medium'] = self.metadata.apply(lambda x: key[x['Medium']], axis=1)

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.root_dir, self.metadata.loc[item, 'Image file'])).convert('RGB')
        label_vector = np.zeros((29,)).astype('float32')
        label_vector[self.metadata.loc[item, 'Medium']] = 1
        return self.transform(image) if self.transform is not None else image, label_vector

    def __len__(self):
        return len(self.metadata.loc[:, 'Image file'])


class MAMe(Data):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.metadata_directory = config['metadata_directory']
        self.images_directory = config['images_directory']
        self.batch_size = config['batch_size']
        if config['version'] == 'full':
            self.metadata_file = 'MAMe_dataset.csv'
        elif config['version'] == 'toy':
            self.metadata_file = 'MAMe_toy_dataset.csv'
        else:
            raise NotImplementedError('Dataset version not implemented')
        self.label_descriptions = {
            row['description']: index
            for index, (_, row) in enumerate(
                pd.read_csv(os.path.join(self.metadata_directory, 'MAMe_labels.csv'), names=["description"],
                            index_col=0).iterrows())
        }

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            dataset=MAMeDataset(metadata_path=os.path.join(self.metadata_directory, self.metadata_file),
                                key=self.label_descriptions, root_dir=self.images_directory, split='train',
                                transform=transforms.ToTensor()),
            shuffle=True, batch_size=self.batch_size
        )

    def get_test_loader(self) -> DataLoader:
        return DataLoader(
            dataset=MAMeDataset(metadata_path=os.path.join(self.metadata_directory, self.metadata_file),
                                key=self.label_descriptions, root_dir=self.images_directory, split='test',
                                transform=transforms.ToTensor()),
            shuffle=True, batch_size=self.batch_size
        )

    def get_val_loader(self) -> DataLoader:
        return DataLoader(
            dataset=MAMeDataset(metadata_path=os.path.join(self.metadata_directory, self.metadata_file),
                                key=self.label_descriptions, root_dir=self.images_directory, split='val',
                                transform=transforms.ToTensor()),
            shuffle=True, batch_size=self.batch_size
        )

    def prepare(self):
        pass

    def get_data_shape(self):
        return [[3, 256, 256], 29]
