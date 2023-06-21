import os, sys, copy, glob
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import Pix2StructProcessor
from utils import custom_round, convert_to_number
import importlib

'''
import importlib
os.chdir('/Users/dhanley/Documents/benetech')
sys.path.append("configs")
sys.path.append("postprocess")
cfg_name = 'cfg_dh_01E'
cfg = importlib.import_module(cfg_name)
cfg = copy.copy(cfg.cfg)
cfg.fold = 0
ds = importlib.import_module(cfg.dataset)
df = pd.read_csv('datamount/train_folded_v01.csv')
test_ds = ds.CustomDataset(df.query('fold==0'), cfg, cfg.val_aug, mode="valid")
val_df = test_ds.df
val_df.shape
val_data_name = glob.glob(f'weights/{cfg_name}/fold0/val*')[0]
val_data = torch.load(val_data_name, map_location=torch.device('cpu'))
pp = importlib.import_module(cfg.post_process_pipeline)
pp_out = pp.post_process_pipeline(cfg, val_data, val_df)
'''


def split_df_to_xy(df):
    df1 = df.copy()
    df2 = df.copy()
    df1['data_series'] = df1.data_series.apply(lambda x: ';'.join(list(map(str, x[0]))))
    df2['data_series'] = df2.data_series.apply(lambda x: ';'.join(list(map(str, x[1]))))
    df1['id'] = df1['id'] + '_x'
    df2['id'] = df2['id'] + '_y'   
    dfout = pd.concat([df1, df2]).set_index('id').sort_index()
    return dfout

def post_process_pipeline(cfg, val_data, val_df):
    
    # Build the prediction dataframe
    break_kv = cfg.break_kv
    break_samp = cfg.break_samp
    
    cfg_name = 'cfg_ch_1'
    cfg = importlib.import_module(cfg_name)
    cfg = copy.copy(cfg.cfg)
    ds = importlib.import_module(cfg.dataset)
    df = pd.read_csv(cfg.train_df)
    test_ds = ds.CustomDataset(df.query(f'fold=={cfg.fold}'), cfg, cfg.val_aug, mode="valid")
    
    chart_data = test_ds.processor.tokenizer.batch_decode(val_data['logits_table'], 
                                                  clean_up_tokenization_spaces = True)
    eos_token = test_ds.processor.tokenizer.eos_token
    chart_data = [s.split(eos_token)[0] for s in chart_data]
    for t in ['<pad>', '<unk>']:
        chart_data = [s.replace(t, '') for s in chart_data]
    chart_data = [[[i.strip() for i in s1.split(break_kv)] for s1 in s.split(break_samp)] for s in chart_data]
    chart_data_min_len = [max(map(len, m)) for m in chart_data]
    chart_data = [[i if len(i)==2 else [np.nan, np.nan] for i in m] for m in chart_data]
    chart_data = [[[i[0] for i in l], [i[1] for i in l]] for l in chart_data]
        
    chart_data = [[convert_to_number(i) for i in s] for s in chart_data]
    
    chart_type = [cfg.chart_map[i] for i in val_data['logits_chart'].argmax(1).tolist()]

    df = pd.DataFrame({'id': val_df.id.tolist(), 'data_series': chart_data, 'chart_type': chart_type})
    pp_out = split_df_to_xy(df)
    
    return pp_out
