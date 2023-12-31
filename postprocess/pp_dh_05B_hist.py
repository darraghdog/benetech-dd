import os, sys, copy, glob
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import Pix2StructProcessor
# from utils import custom_round, convert_to_number
import importlib
import collections

def custom_round(num):
    neg = False 
    if num < 0:
        neg, num = True, -num
    base_val = int(np.floor(np.log10(num)))
    num1 = num / (10 ** (base_val - 2))
    num1 = int(np.round(num1))
    num1 *= (10 ** (base_val - 2))
    #print(num1, base_val)
    if (num1 <10) and (num1>1):
        if num1%1==0:
            num1 = int(num1)
        else:
            num1 = np.round(num1, (-base_val+3))
    elif (num1<1):
        num1 = np.round(num1, (-base_val+3))
    else:
        num1 = int(num1)
    if neg: num1 = -num1
    return num1


def convert_and_fill_mean(l, type_ = int):
    l = [np.nan if str(i).strip() in ['nan', ''] else i for i in l]
    idx_nan = [t for t,i in enumerate(l) if i!=i]
    l = [i for i in l if i==i]
    mout = np.array(l).astype(type_)
    if (len(idx_nan)>0):
        try:
            fill_na = custom_round(mout.mean())
        except:
            fill_na = 0.
        if type_==int: 
            fill_na = int(round(fill_na))
        mout = mout.tolist()
        for pos in idx_nan:
            mout.insert(pos, fill_na)
        return mout

    else:
        mout = mout.tolist()
        return mout


def convert_to_number(l):
    if all([i!=i for i in l]):
        return [0]* len(l)
    try:
        return convert_and_fill_mean(l, type_ = int)
    except ValueError:
        try:
            return convert_and_fill_mean(l, type_ = float)
        except ValueError:
            idx_nan = [t for t,i in enumerate(l) if i!=i]
            l = [i for i in l if i==i][:]
            if len(idx_nan)>0:
                fill_na = l[0]
                for pos in idx_nan:
                    l.insert(pos, fill_na)
            return l

'''
import importlib
os.chdir('/Users/dhanley/Documents/benetech')
sys.path.append("configs")
sys.path.append("postprocess")
sys.path.append("metrics")
cfg_name = 'cfg_dh_05K_icd22_2048_gen'
cfg_name = 'cfg_dh_05K_icd22_2048'
cfg_name = 'cfg_dh_08Q_val'
cfg = importlib.import_module(cfg_name)
cfg = copy.copy(cfg.cfg)
cfg.fold = 0
ds = importlib.import_module(cfg.dataset)
df = pd.read_csv('datamount/train_folded_v03.csv')
test_ds = ds.CustomDataset(df.query('fold==0'), cfg, cfg.val_aug, mode="valid")
val_df = test_ds.df
val_df.shape
val_data_name = glob.glob(f'weights/{cfg_name}/fold0/val*')[0]
val_data = torch.load(val_data_name, map_location=torch.device('cpu'))
pp = importlib.import_module(cfg.post_process_pipeline)
pp_out = pp.post_process_pipeline(cfg, val_data, val_df)
pp_out.shape
from metrics.metric_dh_2p import calc_metric
calc_metric(cfg, pp_out, val_df, pre="val")


pred_lens = pp_out[::2].data_series.str.split(';').apply(len).values
act_lens = val_df.flattened_label.str.split('\\|\\|').str[0].str.split(';').apply(len)
idx = (val_df.source_type=='extracted').values
pd.crosstab((pred_lens[idx]==act_lens[idx]), val_df.chart_type[idx])

# cfg_dh_05K_icd22_2048_gen
chart_type       horizontal_bar  line  scatter  vertical_bar
flattened_label                                             
False                         0    18       26             3
True                         15    67        7            88

# cfg_dh_05K_icd22_2048
chart_type       horizontal_bar  line  scatter  vertical_bar
flattened_label                                             
False                         0    17       16             4
True                         15    68       17            87



Failed on scoring index 2212 gt & pred series and type ([6, 2, 10, 10, 6, 10, 4, 22, 6, 14, 6], 'horizontal_bar', ['6', '2', '10', '15^-20', '6', '10', '4', '22', '6', '14', '6'], 'horizontal_bar')
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

    processor = Pix2StructProcessor.from_pretrained(cfg.backbone, is_vqa = False)
    chart_data =processor.tokenizer.batch_decode(val_data['logits_table'],
                                                  clean_up_tokenization_spaces = True)
    eos_token = processor.tokenizer.eos_token
    chart_data = [s.split(eos_token)[0] for s in chart_data]
    for t in ['<pad>', '<unk>']:
        chart_data = [s.replace(t, '') for s in chart_data]

    chart_data_ctr = [i[:80].count('|') for i in chart_data]

    chart_data = [i.split('|')[-1] for i in chart_data]

    chart_data = [[[i.strip() for i in s1.split(break_kv)] for s1 in s.split(break_samp)] for s in chart_data]
    chart_data_min_len = [max(map(len, m)) for m in chart_data]
    chart_data = [[i if len(i)==2 else [np.nan, np.nan] for i in m] for m in chart_data]
    chart_data = [[[i[0] for i in l], [i[1] for i in l]] for l in chart_data]

    chart_data2 = []
    for s in chart_data:
        res = []
        for j,i in enumerate(s):
            #add if hist here
            #if type is vertical and last y do not convert from placeholder token 
            if (j == 1) and (i[-1] == 'nan') and (len(i)>1):
                res += [convert_to_number(i[:-1])]
            else:
                res += [convert_to_number(i)]
        chart_data2 += [res]
    

    chart_type = [cfg.chart_map[i] for i in val_data['logits_chart'].argmax(1).tolist()]

    df = pd.DataFrame({'id': val_df.id.tolist(), 'data_series': chart_data2, 'chart_type': chart_type})
    pp_out = split_df_to_xy(df)

    return pp_out

