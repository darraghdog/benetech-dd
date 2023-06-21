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

cfg.data_folder = f"/raid/benetech-making-graphs-accessible/train/"
# cfg.mask_folder = f"/raid/scatter_kp_images_ext_v3/masks/"

cfg.train_df = f'/mount/benetech/data/train_icdar22v07/train_folded_v03_icdar22v07.csv'
cfg.icdar_folder = "/raid/train_icdar22v07/"

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
cfg.model = "mdl_ch_4"
cfg.return_logits = False
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.nms_kernel_size = 3
cfg.nms_padding = 1
cfg.n_threshold = 0.5
cfg.pretrained = True
cfg.in_channels = 3
cfg.pool = 'gem'
cfg.gem_p_trainable = False
cfg.return_embeddings = False
#cfg.mixup_beta =1

# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs = 10
cfg.lr = 0.0001 * 8
cfg.optimizer = "AdamW"
cfg.weight_decay = 0.
cfg.clip_grad = 4.
cfg.warmup = 1
cfg.batch_size = 10
cfg.mixed_precision = False # True
cfg.pin_memory = False
cfg.grad_accumulation = 1.
cfg.num_workers = 8


# DATASET
cfg.dataset = "ds_ch_4"
cfg.normalization = "simple"
cfg.chart_map = ['vertical_bar', 'line', 'scatter', 'dot', 'horizontal_bar']
cfg.chart_types = cfg.chart_map
cfg.n_classes = len(cfg.chart_map)
cfg.oof_cutoff = 0.9

#EVAL
cfg.calc_metric = False
cfg.simple_eval = False
# augs & tta

# Postprocess
cfg.post_process_pipeline = 'pp_ch_1'# "pp_ch_sunet_1"
cfg.metric = 'default_metric' # "metric_ch_sunet_1"
# augs & tta

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True

cfg.train_aug = A.Compose([
    A.Resize(448,448),
#     A.HorizontalFlip(p=0.5),
#     A.Transpose(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.5),

    A.RandomCrop(always_apply=False, p=1.0, height=384, width=384), 
#     A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
#     A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.5),
    #A.InvertImg(p=0.5),
#     A.Cutout(num_holes=8, max_h_size=36, max_w_size=36, p=0.8),
])


cfg.val_aug = A.Compose([
    A.Resize(448,448),
#     A.Resize(int(cfg.image_height*1.125),int(cfg.image_width*1.125)),
#     A.PadIfNeeded (min_height=256, min_width=940),
#     A.LongestMaxSize(cfg.image_width_orig,p=1),
#     A.PadIfNeeded(cfg.image_width_orig, cfg.image_height_orig, border_mode=cv2.BORDER_CONSTANT,p=1),
#     A.CenterCrop(always_apply=False, p=1.0, height=cfg.image_height, width=cfg.image_width), 
#     A.Resize(cfg.img_size[0],cfg.img_size[1])
])
