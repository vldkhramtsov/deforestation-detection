import collections

import numpy as np
import pandas as pd
from albumentations import (
    CLAHE, RandomRotate90, Flip, OneOf, Compose, RGBShift, RandomSizedCrop, RandomBrightnessContrast, Transpose, ElasticTransform)
from albumentations.pytorch.transforms import ToTensor
from catalyst.dl.utils import UtilsFactory
from catalyst.data.sampler import BalanceClassSampler
from utils import get_filepath, read_tensor, filter_by_channels


def add_record(data_info, dataset_folder, name, position):
    return data_info.append(
        pd.DataFrame({
            'dataset_folder': dataset_folder,
            'name': name,
            'position': position
        }, index=[0]),
        sort=True, ignore_index=True
    )


class Dataset:
    def __init__(self, channels, dataset_path, image_size, batch_size, num_workers):
        self.channels = channels
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.images_folder = "images"
        self.image_type = "tiff"
        self.masks_folder = "masks"
        self.mask_type = "png"

    def get_input_pair(self, data_info_row):
        if len(self.channels) == 0:
            raise Exception('You have to specify at least one channel.')

        instance_name = '_'.join([data_info_row['name'], data_info_row['position']])
        image_path = get_filepath(
            self.dataset_path, data_info_row['dataset_folder'], self.images_folder,
            instance_name, file_type=self.image_type
        )
        mask_path = get_filepath(
            self.dataset_path, data_info_row['dataset_folder'], self.masks_folder,
            instance_name, file_type=self.mask_type
        )

        images_array = filter_by_channels(
            read_tensor(image_path),
            self.channels
        )

        if images_array.ndim == 2:
            images_array = np.expand_dims(images_array, -1)

        masks_array = read_tensor(mask_path)

        aug = Compose([
            RandomRotate90(),
            Flip(),
            OneOf([
                RandomSizedCrop(
                    min_max_height=(int(self.image_size * 0.7), self.image_size),
                    height=self.image_size, width=self.image_size),
                RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1),
                ElasticTransform(alpha=10, sigma=5, alpha_affine=5)
            ], p=0.75),
            ToTensor()
        ])

        augmented = aug(image=images_array, mask=masks_array)
        augmented_images = augmented['image']
        augmented_masks = augmented['mask']

        return {'features': augmented_images, 'targets': augmented_masks}

    def create_loaders(self, train_df, val_df):
        labels = [(x["mask_pxl"]==0)*1 for x in train_df]
        sampler = BalanceClassSampler(labels, mode="upsampling")
        train_loader = UtilsFactory.create_loader(
            train_df,
            open_fn=self.get_input_pair,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            sampler=sampler)
        
        labels = [(x["mask_pxl"]>5)*1 for x in val_df]
        sampler = BalanceClassSampler(labels, mode="upsampling")
        valid_loader = UtilsFactory.create_loader(
            val_df,
            open_fn=self.get_input_pair,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            sampler=sampler)

        loaders = collections.OrderedDict()
        loaders['train'] = train_loader
        loaders['valid'] = valid_loader

        return loaders
    
    def create_test_loaders(self, test_df):
        test_loader = UtilsFactory.create_loader(
            test_df,
            open_fn=self.get_input_pair,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)

        loaders = collections.OrderedDict()
        loaders['test'] = test_df
        return loaders
