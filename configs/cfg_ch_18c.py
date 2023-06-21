import os
import sys
from importlib import import_module
import platform
import cv2
import json
import numpy as np
import torch
import collections



from default_config import basic_cfg
import pandas as pd

cfg = basic_cfg
cfg.debug = True

# paths

# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"/mount/benetech/models/{os.path.basename(__file__).split('.')[0]}"
cfg.data_dir = f"/raid/benetech-making-graphs-accessible/"
cfg.data_folder = cfg.data_dir + "train/"
cfg.train_df = f'/mount/benetech/data/train_icdar22v07/train_folded_v03_icdar22v07.csv'
cfg.icdar_folder = "/raid/train_icdar22v07/"
cfg.synth_folder = "/raid/dot_v1_converted/"

cfg.pubmed_anno = ['via_project_round_v02__v03_bar.json', 
                   'via_project_round_v01__v04.json', 
                   'via_project_round_v02__v03_linesc.json']

cfg.pubmed_anno = [f'/mount/benetech/data/pubmed/{i}' for i in cfg.pubmed_anno]  
cfg.pubmed_data_dir = f'/raid/round1_v04/'
cfg.pubmed_data_dir2 = f'/raid/'

cfg.pubmed_anno2 = f'/mount/benetech/data/pubmed/via_pseudo_labelled_v07.json'
cfg.pubmed_data_dir2 = f'/raid/round1_v04/'

cfg.pubmed_anno_scatter = ['via_xy_pseudo_v05_single_series_scatter_deplot.json']
cfg.pubmed_anno_scatter = [f'/mount/benetech/data/pubmed/{i}' for i in cfg.pubmed_anno_scatter] 

# stages
cfg.test = False
cfg.test_data_folder = cfg.data_dir + "test/"
cfg.train = True
cfg.train_val =  False
cfg.eval_epochs = 1

#logging
cfg.neptune_project = "light/benetech"
cfg.neptune_connection_mode = "async"
cfg.tags = "base"

#model
cfg.model = "mdl_dh_08O"
cfg.pretrained_weights = '/mount/benetech/models/cfg_ch_15e/fold0/checkpoint_last_seed538039.pth'
cfg.backbone = "google/deplot"
cfg.freeze_from = 'encoder.emb'
cfg.use_bfloat16 = False
#cfg.backbone = "google/flan-t5-base"
cfg.auxiliary_loss = False
cfg.pretrained = True
cfg.in_channels = 3
#cfg.pool = 'gem'
#cfg.gem_p_trainable = False
cfg.return_embeddings = False
#cfg.mixup_beta =1
cfg.is_vqa = False
cfg.max_patches = 2048

# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs = 20
cfg.lr = np.array([3e-5, 1e-4]) * 3
cfg.optimizer = "Adafactor_mixed" # "AdamW_mixed"
cfg.weight_decay = 1e-2
cfg.clip_grad = 4.
cfg.warmup = 1
cfg.batch_size = 2
cfg.mixed_precision = True # True
cfg.pin_memory = False
cfg.grad_accumulation = 4.
cfg.num_workers = 8


# DATASET
cfg.dataset = "ds_dh_08ZK_aug2_dot"
cfg.colors = 'navy blue aqua teal olive green lime yellow orange red maroon fuchsia purple black gray silver'.split(' ')
cfg.curr_epoch = 0
cfg.max_label_length = 384
cfg.max_table_length = 384
cfg.chart_map = ['vertical_bar', 'line', 'scatter', 'dot', 'horizontal_bar']
cfg.dtype_map = ['numerical', 'categorical']
cfg.break_kv = '<extra_id_99>'
cfg.break_samp = '<extra_id_98>'
cfg.break_wt = 2.
cfg.axis_type = {'vertical_bar': ['categorical', 'numerical'], 
                 'horizontal_bar': ['numerical', 'categorical'],
                 'dot': ['both', 'numerical'], 
                 'line': ['categorical', 'numerical'], 
                 'scatter': ['numerical', 'numerical']}
cfg.axis_dependant = {'vertical_bar': [True, False], 
                 'horizontal_bar': [False, True],
                 'dot': [True, False], 
                 'line': [True, False], 
                 'scatter': [True, False]}
cfg.oof_cutoff = 0.9
cfg.round_bins = 100
cfg.round_precision = 2
cfg.raw_num_range = [0, cfg.round_bins] # If the series is in this range, do not to any processing

# Relation ship between number of tick labels and bucketing
cfg.tick_bucketing = collections.OrderedDict([(0, 10), (12, 5) , (20, 2)])


#EVAL
cfg.metric = 'metric_dh_2i'
cfg.calc_metric = True
cfg.simple_eval = False
# augs & tta

# Postprocess
cfg.post_process_pipeline =  "pp_dh_08O"
# augs & tta

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True
cfg.generate = False
if cfg.generate:
    cfg.num_beams=1
    cfg.temperature=1
    cfg.top_k=1
    cfg.train = False
    cfg.train_val = False
    cfg.val = True
    cfg.epochs = 1
    cfg.save_checkpoint = False
    cfg.save_weights_only = False
    cfg.save_only_last_ckpt = False
    cfg.model = "mdl_dh_08I_val"
    cfg.post_process_pipeline =  "pp_dh_05B"
    cfg.oof_cutoff = 0.5
    cfg.batch_size = 32

import albumentations as A

cfg.train_aug = A.Compose([A.PixelDropout(dropout_prob=0.1, p=0.5)
                          ])
cfg.val_aug = A.Compose([])
