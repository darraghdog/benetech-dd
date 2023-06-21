import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2


def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


tr_collate_fn = None
val_collate_fn = None


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.df = df.copy()
        self.mode = mode
        self.ids = self.df["id"].astype(str).values
        
        self.aug = aug
        if mode == "test":
            self.data_folder = cfg.test_data_folder
            
        else:
            self.n = self.df['flattened_label'].apply(lambda x: len(x.split('||')[0].split(';'))).values
            self.data_folder = cfg.data_folder
            self.mask_folder = cfg.mask_folder

    def __getitem__(self, idx):

        img = self.load_one(idx)
        if self.mode == 'test':
            mask = np.zeros_like(img[:,:,0]).astype(float)
            n = 1
        else:
            mask = self.load_mask(idx)
            n = self.n[idx]

        if self.aug:
            img, mask = self.augment(img, mask)

        img = self.normalize_img(img)
        torch_img = torch.tensor(img).float().permute(2,0,1)
        
        
        feature_dict = {
            "input": torch_img,
            "mask": torch.tensor(mask),
            "n":torch.tensor(n).long()
        }
        return feature_dict

    def __len__(self):
        return len(self.ids)
    
    def load_mask(self, idx):
        
        path = self.mask_folder + self.ids[idx] + '.jpg'
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:,:,1] / 255.
        return mask

    def load_one(self, idx):
        path = self.data_folder + self.ids[idx] + '.jpg'
        img = cv2.imread(path)
        return img

    def augment(self, img, mask):
        img = img.astype(np.float32)
        transformed = self.aug(image=img, mask=mask)
        trans_img = transformed["image"]
        trans_mask= transformed["mask"]
        return trans_img, trans_mask

    def normalize_img(self, img):

        if self.cfg.normalization == "image":
            img = (img - img.mean()) / (img.std() + 1e-4)
            img = img.clip(-20, 20)

        elif self.cfg.normalization == "simple":
            img = img / 255

        elif self.cfg.normalization == "min_max":
            img = img - np.min(img)
            img = img / np.max(img)

        return img
