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
        
        self.cfg = cfg
        self.df = df.copy()
        self.df = self.df[self.df['chart_type'].isin(cfg.chart_types)].copy()
        print(self.df['chart_type'].value_counts())
        
        self.df['keep'] = True
        if 'oof_cfg_ch_1e' in self.df.columns:
            idxgen = (self.df.source_type =='generated').values
            idxcut = (self.df['oof_cfg_ch_1e'].values >= cfg.oof_cutoff)
            self.df.loc[(idxgen & idxcut), 'keep'] = [False] * (idxgen & idxcut).sum()
            self.df = self.df[self.df.keep]
        
        self.mode = mode
        if mode == "test":
            self.data_folder = cfg.test_data_folder
        else:
            self.data_folder = cfg.data_folder
        self.df['image_path'] = cfg.data_folder + '/images/' #+ self.df['id'] + '.jpg'
        self.df['anno_path'] = cfg.data_folder + '/annotations/'# + self.df['id'] + '.json'
        
        self.df.loc[self.df.fold==5, 'image_path'] = self.df.loc[self.df.fold==5, 'image_path'].str.replace(cfg.data_folder, cfg.icdar_folder)
        self.df.loc[self.df.fold==5, 'anno_path'] = self.df.loc[self.df.fold==5, 'anno_path'].str.replace(cfg.data_folder, cfg.icdar_folder)
        
#         self.df.loc[self.df.fold==6, 'image_path'] = self.df.loc[self.df.fold==6, 'image_path'].str.replace(cfg.data_folder, cfg.adobe_folder)
#         self.df.loc[self.df.fold==6, 'anno_path'] = self.df.loc[self.df.fold==6, 'anno_path'].str.replace(cfg.data_folder, cfg.adobe_folder)
        
#         self.df.loc[self.df.fold==7, 'image_path'] = self.df.loc[self.df.fold==7, 'image_path'].str.replace(cfg.data_folder, cfg.chart2text_folder)
#         self.df.loc[self.df.fold==7, 'anno_path'] = self.df.loc[self.df.fold==7, 'anno_path'].str.replace(cfg.data_folder, cfg.chart2text_folder)
        
        self.df['image_path'] += self.df['id'] + '.jpg'
        self.df['anno_path'] +=  self.df['id'] + '.json'

#         self.ids = self.df["id"].astype(str).values
        chart_types = self.df['flattened_label'].apply(lambda x: x.split('||')[1]).values
        self.labels = np.array([self.cfg.chart_map.index(chart_type) for chart_type in chart_types])
        self.aug = aug

        if mode == "test":
            self.data_folder = cfg.test_data_folder
            
        else:
            self.data_folder = cfg.data_folder
#             self.mask_folder = cfg.mask_folder

    def __getitem__(self, idx):

        img = self.load_one(idx)
        label = self.labels[idx]
        if self.aug:
            img = self.augment(img)

        img = self.normalize_img(img)
        torch_img = torch.tensor(img).float().permute(2,0,1)
        
        feature_dict = {
            "input": torch_img,
            "target":torch.tensor(label).long()
        }
        return feature_dict

    def __len__(self):
        return len(self.df)
    
    def load_mask(self, idx):
        
        path = self.mask_folder + self.ids[idx] + '.jpg'
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:,:,1] / 255.
        return mask

    def load_one(self, idx):
        path = self.df.iloc[idx]['image_path']
        img = cv2.imread(path)
        return img

    def augment(self, img):
        img = img.astype(np.float32)
        transformed = self.aug(image=img)
        trans_img = transformed["image"]
        return trans_img

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
