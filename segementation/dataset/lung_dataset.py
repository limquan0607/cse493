import os
import numpy as np
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import torch


class LungTumorDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, masks_dir, transform=None):
        self.data_dir = data_dir
        self.masks_dir = masks_dir
        self.all_files = os.listdir(self.data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def augment(self, data, mask):
        # Fix to lack of randomness problem when using something other than pytorch
        rng = np.random.default_rng()
        random_seed = rng.integers(0, 1000000)
        ia.seed(random_seed)

        mask = SegmentationMapsOnImage(mask, shape=mask.shape)
        if self.transform == None:
            return data, mask
        aug_data, aug_mask = self.transform(image=data, segmentation_maps=mask)
        return aug_data, aug_mask.get_arr()

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.all_files[idx])
        mask_path = os.path.join(self.masks_dir, self.all_files[idx])
        data = np.load(data_path)
        mask = np.load(mask_path)
        if self.transform:
            # mask = mask.astype(np.uint8)
            # data = data.astype(np.uint8)
            data, mask = self.augment(data, mask)
        data, mask = np.expand_dims(data, axis=0), np.expand_dims(mask, axis=0)
        return data.astype(np.float32), mask.astype(np.float32)


def get_dataset(preprocessed_input_dir, aug_pipeline=None, data_type="train"):
    data_dir = os.path.join(preprocessed_input_dir, data_type, "data")
    label_dir = os.path.join(preprocessed_input_dir, data_type, "mask")
    return LungTumorDataset(data_dir, label_dir, transform=aug_pipeline)


def get_all_datasets(preprocessed_input_dir, aug_pipeline=None):
    train_dataset = get_dataset(preprocessed_input_dir, aug_pipeline, "train")
    val_dataset = get_dataset(preprocessed_input_dir, None, "val")
    print(
        "---------------------------------------------------------------------------------------------------------------"
    )
    print("Loading dataset")
    print(
        f"There are {len(train_dataset)} train images and {len(val_dataset)} val images"
    )
    print(
        "---------------------------------------------------------------------------------------------------------------"
    )
    return train_dataset, val_dataset
