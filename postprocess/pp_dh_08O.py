import os, sys, copy, glob
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import Pix2StructProcessor
from utils import custom_round, convert_to_number
import importlib
import collections

'''
import importlib
os.chdir('/Users/dhanley/Documents/benetech')
sys.path.append("configs")
sys.path.append("postprocess")
sys.path.append("metrics")
cfg_name = 'cfg_dh_08I'
cfg = importlib.import_module(cfg_name)
cfg = copy.copy(cfg.cfg)
cfg.fold = 0
ds = importlib.import_module(cfg.dataset)
df = pd.read_csv('datamount/train_folded_v03.csv')
test_ds = ds.CustomDataset(df.query('fold==0'), cfg, cfg.val_aug, mode="valid")
val_df = test_ds.df
val_df.flattened_label_all = val_df.flattened_label.copy()
val_df.shape
val_data_name = glob.glob(f'weights/{cfg_name}/fold0/val*')[0]
val_data = torch.load(val_data_name, map_location=torch.device('cpu'))
cfg.post_process_pipeline = 'pp_dh_08I'
pp = importlib.import_module(cfg.post_process_pipeline)
pp_out = pp.post_process_pipeline(cfg, val_data, val_df)
pp_out.shape
from metrics.metric_dh_2i import calc_metric
calc_metric(cfg, pp_out, val_df, pre="val")

calc_metric(cfg, pp_out_ls[0], val_df, pre="val")

pp_out1 = pp_out_ls[1].copy()
pp_out1. data_series = pp_out1. data_series.str.split(';').apply(lambda i: i[::-1]).str.join(';')
calc_metric(cfg, pp_out1, val_df, pre="val")




val_df.flattened_label = \
val_df.flattened_label.str.split('\\|\\|').str[0].str.split('~~').str[0] + '||' + \
val_df.flattened_label.str.split('\\|\\|').str[1] + '||' + \
val_df.flattened_label.str.split('\\|\\|').str[2] + '||' + \
val_df.flattened_label.str.split('\\|\\|').str[3].str.split('~~').str[0] + '||' + \
val_df.flattened_label.str.split('\\|\\|').str[4] + '||' + \
val_df.flattened_label.str.split('\\|\\|').str[5]

val_df.flattened_label


iddx = (val_df.source_type=='extracted').values
act_len = val_df.flattened_label.str.split('\\|\\|').str[0].str.split(';').apply(len)
pred_lens = np.stack((pp_out_ls[0].data_series.str.split(';').apply(len)[::2].values,
                      pp_out_ls[1].data_series.str.split(';').apply(len)[::2].values,
                      pp_out_ls[2].data_series.str.split(';').apply(len)[::2].values,
                      pp_out_ls[3].data_series.str.split(';').apply(len)[::2].values))

matches = pred_lens == act_len.values[None,:]



val_df['match'] = matches.sum(0)


pd.crosstab(val_df[iddx].chart_type, val_df[iddx].match>1)
pd.crosstab(val_df[iddx].chart_type, val_df[iddx].match0)
pd.crosstab(val_df[iddx].chart_type, val_df[iddx].match1)
pd.crosstab(val_df[iddx].chart_type, val_df[iddx].match2)


pd.crosstab(val_df[iddx].chart_type, np.floor(np.median(pred_lens, 0)) == act_len)

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
    
    logit_cols = 'logits_table_1 logits_table_2'.split()
    
    chart_type = [cfg.chart_map[i] for i in val_data['logits_chart'].argmax(1).tolist()]
    pp_out_ls = []
    for logits_table in logit_cols:
        chart_data =processor.tokenizer.batch_decode(val_data[logits_table],
                                                      clean_up_tokenization_spaces = True)
        eos_token = processor.tokenizer.eos_token
        chart_data = [s.split(eos_token)[0] for s in chart_data]
        for t in ['<pad>', '<unk>']:
            chart_data = [s.replace(t, '') for s in chart_data]
    
        # chart_data_ctr = [i[:80].count('|') for i in chart_data]
    
        chart_data = [i.split('|')[-1] for i in chart_data]
    
        chart_data = [[[i.strip() for i in s1.split(break_kv)] for s1 in s.split(break_samp)] for s in chart_data]
        chart_data_min_len = [max(map(len, m)) for m in chart_data]
        chart_data = [[i if len(i)==2 else [np.nan, np.nan] for i in m] for m in chart_data]
        chart_data = [[[i[0] for i in l], [i[1] for i in l]] for l in chart_data]
    
        chart_data = [[convert_to_number(i) for i in s] for s in chart_data]
        
        df = pd.DataFrame({'id': val_df.id.tolist(), 'data_series': chart_data, 'chart_type': chart_type})
        pp_out = split_df_to_xy(df)
        pp_out_ls.append(pp_out.copy())
    
    return pp_out_ls[0]

