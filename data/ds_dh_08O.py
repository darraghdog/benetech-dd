import platform
import os
import math
import torch
import numpy as np
from tqdm import tqdm
from bounding_box import bounding_box as bbfn
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from PIL import Image
import json
from tqdm import tqdm
import pandas as pd
import re
from PIL import Image
import imagesize
from transformers import Pix2StructProcessor
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any
import decimal
#from utils import benetech_score
#from utils import set_pandas_display
from operator import itemgetter



def is_numerical(x):
    """Test whether input can be parsed as a Python float."""
    try:
        float(x)
        return True
    except ValueError:
        return False
    
def round_numerical(l, cfg):
    
    drop_series = False
    
    # Fill na first, eg. 60923b97d2b5.json
    if any([i!=i for i in l]):
        mean_val = np.nanmean(l)
        if isinstance([i for i in l if  i==i][0], int):
            mean_val = int(round(mean_val))
        l = [i if i==i else mean_val for i in l]
            
    rng_min = min(l)
    rng_max = max(l)
    rng = rng_max - rng_min
    
    if rng==0:
        # extreme case, drop it
        drop_series = True
        return l, drop_series
    
    all_ints = all([isinstance(i, int) for i in l])
    
    if (rng_min >= cfg.raw_num_range[0]) and (rng_max <= cfg.raw_num_range[1]) and all_ints:
        return list(map(str, l)), drop_series
    
    # Precision of rounding. cfg.round_precision how many digits to fill
    bin_num = rng / cfg.round_bins
    base_val = int(np.floor(np.log10(rng)))
    base_round = base_val - cfg.round_precision
    
    bin_num = 10 ** base_round 
    lout = [i - i%bin_num for i in l]
    
    # If we have no decimals, change all to int
    if all([(i%1)==0 for i in lout]):
        lout  = list(map(int, lout))
        lout  = list(map(str, lout))
        return list(map(str, lout)), drop_series
    
    # If we have decimals, change all to same precision
    round_dec = -base_val
    lout = [f'{i:.{1 + round_dec}f}' for i in lout]
    
    return lout, drop_series

def is_monotonic(s):
    diffs =  s[1:] - s[:-1]
    if all(x >= 0 for x in diffs) or all(x <= 0 for x in diffs):
        return True
    return False

# _data_series, _labels = y_data_series, y_labels[::-1]
def bucket_in_label(_data_series, _labels, cfg):
    if len(_labels)<2:
        return _data_series, False
    _labels = [i.replace(",", "").replace("%", "") for i in _labels]
    try:
        ylbls = np.array(list(map(int, _labels)))
    except:
        try:
            ylbls = np.array(list(map(float, _labels)))
        except:
            return _data_series, False
    if not is_monotonic(ylbls):
        return _data_series, False
    
    n_buckets = min(cfg.tick_bucketing.values())
    for k,v in cfg.tick_bucketing.items():
        if len(ylbls)>k: 
            n_buckets = v
    
    bucket_size = (max(ylbls) - min(ylbls)) / ((len(ylbls)-1) * n_buckets)
    if bucket_size > 0.5: 
        bucket_size = max(int(round(bucket_size)), 1)
        buckets = np.arange(min(ylbls), max(ylbls)+1, bucket_size)
        _data_series = [buckets[(abs(buckets-float(i))).argmin()] for i in _data_series]
        _data_series = list(map(int, _data_series))
        _data_series = list(map(str, _data_series))
        return _data_series,  True
    else:
        return _data_series,  False

def load_annotation(file_path: str):
    with open(file_path) as annotation_f:
        ann_example = json.load(annotation_f)
    return ann_example

'''
              bucketed
type                  
dot           0.000000
line          0.575747
scatter       0.575989
vertical_bar  0.739361

              bucketed
type                  
dot               1108
line              5888
scatter           3007
vertical_bar      4723
'''

def extract_data3(file_path, cfg):
    
    # ll = []
    # for file_path in filenames[:]:
    
    #if '820d4d58c14b' in file_path: break
    axis_type = cfg.axis_type
    annotation_example = load_annotation(file_path)
    
    # Extracting the file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    chart_type = annotation_example['chart-type']

    chart_id_x = f'{file_name}_x'
    chart_id_y = f'{file_name}_y'

    x_val_xtype = annotation_example['axes']['x-axis']['values-type']
    y_val_xtype = annotation_example['axes']['y-axis']['values-type']
    
    x_data_series = [item['x'] for item in annotation_example['data-series']]
    y_data_series = [item['y'] for item in annotation_example['data-series']]  
    
    if chart_type == 'scatter':
        sort_key = np.lexsort(( list(map(float, y_data_series)), \
                                list(map(float, x_data_series))  ))
        x_data_series = [x_data_series[i] for i in sort_key]
        y_data_series = [y_data_series[i] for i in sort_key]
    
    if chart_type == 'horizontal_bar':
        sort_y_idx = np.array(list(map(float, x_data_series))).argsort()
    else:
        sort_y_idx = np.array(list(map(float, y_data_series))).argsort()
    
    id_texts = {d['id']:d['text'] for d in annotation_example['text']}
    x_labels = [id_texts[d['id']] for d in annotation_example['axes']['x-axis']['ticks']]
    y_labels = [id_texts[d['id']] for d in annotation_example['axes']['y-axis']['ticks']]
    if 'horizontal_bar'==chart_type:
        x_labels, y_labels = y_labels, x_labels 
    
    x_dtype, y_dtype = axis_type[chart_type]
    
    drop_series_y = False
    if y_dtype=='numerical':
        y_data_series, bucketed = bucket_in_label(y_data_series, y_labels, cfg)
        if len(set(y_data_series))==1: drop_series_y = True
        if not bucketed:
            y_data_series, drop_series_y = round_numerical(y_data_series, cfg)
        
        '''
        print(file_name)
        print(y_data_series)
        print('')
        '''
        
    else:
        assert all([isinstance(i, str) for i in y_data_series])
    
    drop_series_x = False
    if x_dtype=='numerical':
        x_data_series, bucketed = bucket_in_label(x_data_series, x_labels, cfg)
        if len(set(x_data_series))==1: drop_series_x = True
        if not bucketed:
            x_data_series, drop_series_x = round_numerical(x_data_series, cfg)
    else:
        assert all([isinstance(i, str) for i in x_data_series])
        
    x_data_series_1 = ';'.join(map(str, x_data_series))
    y_data_series_1 = ';'.join(map(str, y_data_series))
    x_data_series_2 = ';'.join(map(str, x_data_series[::-1]))
    y_data_series_2 = ';'.join(map(str, y_data_series[::-1]))
    
    x_data_series_3 = ';'.join(map(str, [x_data_series[i] for i in sort_y_idx]))
    y_data_series_3 = ';'.join(map(str, [y_data_series[i] for i in sort_y_idx]))
    x_data_series_4 = ';'.join(map(str, [x_data_series[i] for i in sort_y_idx[::-1]]))
    y_data_series_4 = ';'.join(map(str, [y_data_series[i] for i in sort_y_idx[::-1]]))

    x_data_series  = f'{x_data_series_1}~~{x_data_series_2}~~{x_data_series_3}~~{x_data_series_4}'
    y_data_series  = f'{y_data_series_1}~~{y_data_series_2}~~{y_data_series_3}~~{y_data_series_4}'

    extracted_data = {
        chart_id_x: {
            'data_series': x_data_series,
            'chart_type': chart_type,
            'val_dtype': x_val_xtype,
            'drop_series': drop_series_x,
            'sort_y_idx': sort_y_idx,
        },
        chart_id_y: {
            'data_series': y_data_series,
            'chart_type': chart_type,
            'val_dtype': y_val_xtype,
            'drop_series': drop_series_y,
            'sort_y_idx': sort_y_idx,
        }
    }

    return extracted_data

# filenames, axis_type = self.df['anno_path'].tolist(), cfg.axis_type
def create_dataframe_from_directory3(filenames, cfg):
    
    axis_type = cfg.axis_type
    data = []
     
    for file_path in tqdm(filenames):
        file_data = extract_data3(file_path, cfg)
        
        for key, value in file_data.items():
            data.append({
                'id': key,
                'data_series': value['data_series'],
                'chart_type': value['chart_type'],
                'val_dtype': value['val_dtype'],
                'drop_series': value['drop_series'],
                'sort_y_idx': value['sort_y_idx'],
            })

    df = pd.DataFrame(data)
    return df

def batch_to_device(batch, device):
    
    batch = {k:v.to(device) for k,v in batch.items()}
    
    return batch

def collate_fn(batch):
    
    keys = batch[0].keys()
    batch_out = {}
    for k in batch[0].keys():
        if 'series_label' not in k:
            batch_out[k] = torch.stack([b[k] for b in batch])
    for tt in range(2):    
        if f'series_label_{tt+1}' in batch[0].keys():
            batch_out[f'series_label_{tt+1}'] = pad_sequence([b[f'series_label_{tt+1}'] for b in batch], padding_value=-100)
    #batch_out['series_mask'] = pad_sequence([torch.ones(len(b['series_label'])).long() for b in batch])
    
    return batch_out

tr_collate_fn = collate_fn
val_collate_fn = collate_fn

'''
df = pd.read_csv(cfg.train_df)#.query('fold==0')

mode="train"
class self:
    1
self = CustomDataset(df, cfg, aug = cfg.train_aug, mode = 'train')
idx = 10

# self = CustomDataset(df, cfg, aug = cfg.val_aug, mode = 'valid')
batch = [self.__getitem__(i) for i in range(0, 200, 199)]
batch = tr_collate_fn(batch)
batch = batch_to_device(batch, 'cpu')
'''


# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5")

class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):

        self.cfg = cfg
        self.df = df.copy()
        
        if 'oof_cfg_ch_1e' in self.df.columns:
            self.df['keep'] = True
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
        self.df['image_path'] += self.df['id'] + '.jpg'
        self.df['anno_path'] +=  self.df['id'] + '.json'
        
        idx = 0
        self.processor = Pix2StructProcessor.from_pretrained(cfg.backbone, is_vqa = cfg.is_vqa)
                
        if not self.mode == 'test':
            gtdf = create_dataframe_from_directory3(self.df['anno_path'].tolist(), cfg)
            gtdf = gtdf.sort_values('id').reset_index(drop = True)
            if mode=="train":
                # Drop series where we specified in functions above
                drop_idx = np.repeat(gtdf.drop_series[::2].values | gtdf.drop_series[1::2].values, 2)
                gtdf = gtdf[~drop_idx].reset_index(drop = True)
            gtdf['key'] = gtdf['id'].str.replace('_x', '').str.replace('_y', '')
            gtdf['flattened_label'] = gtdf['data_series'] + '||' + gtdf['chart_type'] + '||' + gtdf['val_dtype']
            gtdf = gtdf.groupby(['key'])['flattened_label'].apply(lambda x: '||'.join(x))

            # Message on dropping
            if mode=="train":
                print(f'Dropping {drop_idx[::2].sum()} of {len(drop_idx[::2])} samples')
                self.df = self.df[self.df.id.isin(gtdf.index.tolist())]

            self.ground_truth  = self.df[['id', 'flattened_label', 'chart_type']].copy()
            self.df['flattened_label']  = gtdf.loc[self.df.id.tolist()].values
        
        # Find the none values and drop them
        self.prompt = 'Extract a table from the chart:'
        self.break_kv = cfg.break_kv 
        self.break_samp = cfg.break_samp 
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # for idx in range(len(self.df)):
        samp = self.df.iloc[idx]
        image = cv2.imread(samp.image_path)
        
        
        encoding = {}
        encoding = self.processor(images=image, text=self.prompt, return_tensors="pt", max_patches = self.cfg.max_patches)
        
        if not self.mode == 'test':
            x_seq, chart_type, x_val_dtype, y_seq, _, y_val_dtype = samp.flattened_label.split('||')
            for tt, (x_s, y_s) in enumerate(zip(x_seq.split('~~'), y_seq.split('~~'))):
                label_pairs = [f'{i}{self.break_kv}{j}' for i,j in zip(x_s.split(';'), y_s.split(';'))]
                label_pairs = f'{self.break_samp}'.join(label_pairs)
                label = self.processor.tokenizer(label_pairs, max_length=self.cfg.max_label_length, truncation=True, return_tensors="pt")
                # self.processor.tokenizer.decode(label.input_ids[0])
                encoding[f'series_label_{tt+1}'] = label['input_ids'][0]
                if tt>1: continue

            encoding['chart_label'] = torch.tensor(self.cfg.chart_map.index(chart_type))
            encoding['xy_dtype']    = torch.tensor([self.cfg.dtype_map.index(x_val_dtype), 
                                                    self.cfg.dtype_map.index(y_val_dtype)])
        
                
        return encoding


def competition_format(df, cfg, gt = False):
    '''
    Convert to competition format to check the preprocessing does not have major issues vs original groundtruth
    '''
    if not gt:
        df['data_series_x'] = df['flattened_label'].apply(lambda x: x.split('||'), 1).str[:2].str.join('||')#.iloc[0]
        df['data_series_y'] = df['flattened_label'].apply(lambda x: x.split('||'), 1).str[3:-1].str.join('||')#.iloc[0]
    else:
        df['data_series_x'] = df['flattened_label'].apply(lambda x: x.split('||'), 1).str[:2].str.join('||')#.iloc[0]
        df['data_series_y'] = df['flattened_label'].apply(lambda x: x.split('||'), 1).str[2:].str.join('||')#.iloc[0]
    d1 = df['id data_series_x'.split()]
    d2 = df['id data_series_y'.split()]
    d1.loc[:, 'id'] = d1['id'].apply(lambda x: f'{x}_x').tolist()
    d2.loc[:, 'id'] = d2['id'].apply(lambda x: f'{x}_y').tolist()
    d1  = d1.rename(columns={"data_series_x": "data_series"})
    d2  = d2.rename(columns={"data_series_y": "data_series"})
    dd = pd.concat([d1, d2], axis = 0)
    dd[['data_series', 'chart_type']] = dd.data_series.apply(lambda x: x.split('||'), 1).tolist()
    
    for ctyp, axis_types in cfg.axis_type.items():
        for ax, dtype in zip(['_x', '_y'], axis_types):
            idx = (dd.chart_type==ctyp).values & (dd.id.str[-2:] == ax).values
            if dtype=='numerical':
                dd.loc[idx, 'data_series'] = dd.loc[idx, 'data_series'].str.split(';').apply(lambda i: list(map(float, i)))
            elif dtype=='categorical':
                dd.loc[idx, 'data_series'] = dd.loc[idx, 'data_series'].str.split(';')
            elif dtype=='both':
                dd.loc[idx, 'data_series'] = dd.loc[idx, 'data_series'].str.split(';')
                idx2 = idx.copy() # index for the numerical dots
                idx2[idx] = dd.loc[idx, 'data_series'].apply(lambda i: all([is_numerical(x) for x in i])).values
                dd.loc[idx2, 'data_series'] = dd.loc[idx2, 'data_series'].apply(lambda i: list(map(float, i)))
    return dd

'''
self.predictions =  self.df[['id', 'flattened_label', 'chart_type']].copy()

(self.ground_truth['flattened_label'] == self.predictions['flattened_label']).mean()
predictions = competition_format(self.predictions, cfg).set_index('id')
ground_truth = competition_format(self.ground_truth, cfg, gt = True).set_index('id')

idx_non_nan = ground_truth.data_series.apply(lambda i: all([j==j for j in i])).values

scores = benetech_score(ground_truth[idx_non_nan], predictions[idx_non_nan], raw_scores = True)
scores.mean()

scores[scores<0.0001]

'''
