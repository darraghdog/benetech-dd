import os
import sys
from importlib import import_module
import platform
import cv2
import json
import numpy as np
import torch
import pandas as pd
import albumentations as A


# sys.path.append("configs")
# sys.path.append("augs")
# sys.path.append("models")
# sys.path.append("data")
# sys.path.append("postprocess")

from default_config import basic_cfg


cfg = basic_cfg
cfg.debug = True

# paths

cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"/mount/benetech/models/{os.path.basename(__file__).split('.')[0]}"

cfg.data_folder = f"/raid/scatter_kp_images_v7b_768/images/"
cfg.mask_folder = f"/raid/scatter_kp_images_v7b_768/masks/"
cfg.train_df = f'/mount/benetech/data/synthetic/scatter_kp_images_v7b_768.csv'

# stages
cfg.test = False
cfg.test_data_folder = cfg.data_folder
cfg.train = True
cfg.train_val =  False
cfg.eval_epochs = 1

#logging
cfg.neptune_project = "light/benetech"
cfg.neptune_connection_mode = "async"
cfg.tags = "base"

#model
cfg.model = "mdl_ch_sunet_2"
cfg.return_logits = False
cfg.backbone = "tf_efficientnet_b7_ns"
cfg.pretrained_weights = ['/mount/benetech/models/cfg_ch_sunet_9a/fold0/checkpoint_last_seed483907.pth',
                          '/mount/benetech/models/cfg_ch_sunet_9a/fold1/checkpoint_last_seed196802.pth',
                          '/mount/benetech/models/cfg_ch_sunet_9a/fold2/checkpoint_last_seed189944.pth',
                          '/mount/benetech/models/cfg_ch_sunet_9a/fold3/checkpoint_last_seed971408.pth',
                          '/mount/benetech/models/cfg_ch_sunet_9a/fold4/checkpoint_last_seed445410.pth'
                         
                         
                         
                         ]
cfg.nms_kernel_size = 3
cfg.nms_padding = 1
cfg.n_threshold = 0.5
cfg.pretrained = True
cfg.in_channels = 3
#cfg.pool = 'gem'
#cfg.gem_p_trainable = False
# cfg.return_embeddings = False
#cfg.mixup_beta =1

# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs = 5
cfg.lr = 0.0001 * 8
cfg.optimizer = "AdamW"
cfg.weight_decay = 0.
cfg.clip_grad = 4.
cfg.warmup = 1
cfg.batch_size = 5
cfg.mixed_precision = False # True
cfg.pin_memory = False
cfg.grad_accumulation = 2.
cfg.num_workers = 8


# DATASET
cfg.dataset = "ds_ch_sunet_4"
cfg.normalization = "simple"

#EVAL
cfg.calc_metric = True
cfg.simple_eval = False
# augs & tta

# Postprocess
cfg.post_process_pipeline =  "pp_ch_sunet_1"
cfg.metric = "metric_ch_sunet_3"
# augs & tta

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True

cfg.train_aug = A.Compose([
#     A.Resize(int(cfg.image_height*1.125),int(cfg.image_width*1.125)),
#     A.HorizontalFlip(p=0.5),
#     A.Transpose(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.5),

    A.RandomCrop(always_apply=False, p=1.0, height=672, width=672), 
#     A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
#     A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.5),
    #A.InvertImg(p=0.5),
#     A.Cutout(num_holes=8, max_h_size=36, max_w_size=36, p=0.8),
])


cfg.val_aug = A.Compose([
#     A.Resize(int(cfg.image_height*1.125),int(cfg.image_width*1.125)),
#     A.PadIfNeeded (min_height=256, min_width=940),
#     A.LongestMaxSize(cfg.image_width_orig,p=1),
#     A.PadIfNeeded(cfg.image_width_orig, cfg.image_height_orig, border_mode=cv2.BORDER_CONSTANT,p=1),
#     A.CenterCrop(always_apply=False, p=1.0, height=cfg.image_height, width=cfg.image_width), 
#     A.Resize(cfg.img_size[0],cfg.img_size[1])
])
