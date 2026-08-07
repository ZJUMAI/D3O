import os
import random

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ordinalclip.utils.logging import get_logger, print_log

from .utils import get_transforms

logger = get_logger(__name__)
print = lambda x: print_log(x, logger=logger)

# Manually set 10 samples to track during training.
ADIENCE_TRACK_SAMPLE_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class RegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        logger,
        train_images_root,
        val_images_root,
        test_images_root,
        train_data_file,
        val_data_file,
        test_data_file,
        transforms_cfg=None,
        train_dataloder_cfg=None,
        eval_dataloder_cfg=None,
        fold=0,
    ):
        super().__init__()
        train_transforms, eval_transforms = get_transforms(**transforms_cfg)

        self.train_set = RegressionDataset(
            logger, train_images_root, train_data_file, "train", fold, train_transforms
        )
        self.val_set = RegressionDataset(
            logger, val_images_root, val_data_file, "valid", fold, eval_transforms
        )
        self.test_set = RegressionDataset(
            logger, test_images_root, test_data_file, "valid", fold, eval_transforms
        )

        self.train_dataloder_cfg = train_dataloder_cfg
        self.eval_dataloder_cfg = eval_dataloder_cfg

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, **self.train_dataloder_cfg)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, **self.eval_dataloder_cfg)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, **self.eval_dataloder_cfg)

class RegressionDataset(Dataset):
    def __init__(self, logger, images_root, data_file, mode, fold, transforms=None):
        self.images_root = images_root
        self.labels = []
        self.images_file = []
        self.transforms = transforms

        root = data_file
        if mode == 'train':
            file_name = ['age_train.txt']
        elif mode == 'valid':
            file_name = ['age_test.txt']

        print("Use the data fold: {}".format(fold))

        for each in file_name:
            f_path = root + '/test_fold_is_' + str(fold) + '/' + each
            with open(f_path, 'r') as f:
                for line in f:
                    splits = line.split()
                    assert len(splits) == 2
                    labels = splits[-1]
                    self.labels.append([int(label) for label in labels])
                    self.images_file.append(splits[0])

        self.label_num = [0, 0, 0, 0, 0, 0, 0, 0]
        for each in self.labels:
            self.label_num[each[0]] += 1

        self.name = mode
        if "val" in self.name or "test" in self.name:
            print(f"Dataset prepare: val/test data_file")
        elif "train" in self.name:
            print(f"Dataset prepare: train data_file")
        else:
            raise ValueError(f"Invalid data_file: {data_file}")

        logger.info(f"Dataset prepare: num of labels: {self.label_num}")
        logger.info(f"Dataset prepare: len of dataset: {len(self.labels)}")

        self.track_sample_indices = [i for i in ADIENCE_TRACK_SAMPLE_INDICES if 0 <= i < len(self.labels)]

    def __getitem__(self, index):
        img_file, target_list = self.images_file[index], self.labels[index]
        if "val" in self.name or "test" in self.name:
            target = target_list[len(target_list) // 2]
        else:
            target = random.choice(target_list)

        full_file = os.path.join(self.images_root, img_file)
        img = Image.open(full_file).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def get_tracking_samples(self):
        images, labels = [], []
        for idx in self.track_sample_indices:
            img, y = self[idx]
            images.append(img)
            labels.append(int(y))

        if len(images) == 0:
            return None

        return {
            "indices": list(self.track_sample_indices),
            "images": torch.stack(images, dim=0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sub_classes": None,
        }

    def __len__(self):
        return len(self.labels)
