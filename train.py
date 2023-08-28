import os
import glob
import gc
from copy import copy
import numpy as np
import pandas as pd
import importlib
import sys
from tqdm import tqdm, notebook
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import transformers
from decouple import Config, RepositoryEnv
import neptune
from neptune.utils import stringify_unsupported
from utils import calc_grad_norm

def get_optimizer(model, cfg):

    params = model.parameters()
    params = [
        {
            "params": [
                param for name, param in model.named_parameters() if "backbone" in name
            ],
            "lr": cfg.lr[0],
        },
        {
            "params": [
                param for name, param in model.named_parameters() if not "backbone" in name
            ],
            "lr": cfg.lr[1],
        },
    ]
    optimizer = Adafactor(params, lr=cfg.lr[1], weight_decay=cfg.weight_decay, scale_parameter=False, relative_step=False)

    return optimizer

BASEDIR= './'#'../input/asl-fingerspelling-config'
for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

env_config = Config(RepositoryEnv('.env'))


parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename", default="cfg_dh_61g3")
parser_args, other_args = parser.parse_known_args(sys.argv)

cfg = copy(importlib.import_module(parser_args.config).cfg)

# Import experiment modules
post_process_pipeline = importlib.import_module(cfg.post_process_pipeline).post_process_pipeline
calc_metric = importlib.import_module(cfg.metric).calc_metric
Net = importlib.import_module(cfg.model).Net
CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device

# Start neptune
fns = [parser_args.config] + [getattr(cfg, s) for s in 'dataset model metric post_process_pipeline'.split()]
fns = sum([glob.glob(f"{BASEDIR }/*/{fn}.py") for fn in  fns], [])
neptune_run = neptune.init_run(
        project=env_config.get("NEPTUNE_PROJECT_NAME"),
        tags="demo",
        mode="async",
        api_token=env_config.get("NEPTUNE_ASL_TOKEN"),
        capture_stdout=False,
        capture_stderr=False,
        source_files=fns
    )
print(f"Neptune system id : {neptune_run._sys_id}")
print(f"Neptune URL       : {neptune_run.get_url()}")
print(f"Target checkpoint : " +  f"checkpoint_{neptune_run._sys_id}.pth")
neptune_run["cfg"] = stringify_unsupported(cfg.__dict__)


# Read our training data
df = pd.read_csv('train_v2_score38_oof.csv')
train_df = df[df["fold"] != 0]
val_df = df[df["fold"] == 0]

cfg.data_folder = "datamount/train_landmarks_v3_mount/"
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set up the dataset and dataloader
train_dataset = CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train", symmetry_fn = 'datamount/symmetry.csv')
val_dataset = CustomDataset(val_df, cfg, aug=cfg.train_aug, mode="val", symmetry_fn = 'datamount/symmetry.csv')
train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,#//4,
        pin_memory=cfg.pin_memory,
        collate_fn=tr_collate_fn,
    )
val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,#//4,
        pin_memory=cfg.pin_memory,
        collate_fn=val_collate_fn,
    )

# Set up the model
model = Net(cfg).to(cfg.device)

total_steps = len(train_dataset)
optimizer = get_optimizer(model)
# optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size),
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size),
            num_cycles=0.5
        )
scaler = GradScaler()


# Start the training and validation loop
cfg.curr_step = 0
optimizer.zero_grad()
total_grad_norm = None    
total_grad_norm_after_clip = None
for epoch in range(cfg.epochs):
    
    cfg.curr_epoch = epoch
    progress_bar = tqdm(range(len(train_dataloader))[:], desc=f'Train epoch {epoch}')
    tr_it = iter(train_dataloader)
    losses = []
    gc.collect()

    model.train()
    for itr in progress_bar:
        cfg.curr_step += cfg.batch_size
        data = next(tr_it)
        torch.set_grad_enabled(True)
        batch = batch_to_device(data, cfg.device)
        with autocast():
            output_dict = model(batch)
        loss = output_dict["loss"]
        losses.append(loss.item())

        if cfg.grad_accumulation >1:
            loss /= cfg.grad_accumulation

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        total_grad_norm = calc_grad_norm(model.parameters())                              
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        total_grad_norm_after_clip = calc_grad_norm(model.parameters())
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        loss_names = [key for key in output_dict if 'loss' in key]
        for l in loss_names:
            neptune_run[f"train/{l}"].log(value=output_dict[l].item(), step=cfg.curr_step)

        neptune_run["lr"].log(
                value=optimizer.param_groups[0]["lr"], step=cfg.curr_step
            )
        if total_grad_norm is not None:
            neptune_run["total_grad_norm"].log(value=total_grad_norm.item(), step=cfg.curr_step)
            neptune_run["total_grad_norm_after_clip"].log(value=total_grad_norm_after_clip.item(), step=cfg.curr_step)
    

    if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
        model.eval()
        torch.set_grad_enabled(False)
        val_data = defaultdict(list)
        val_score =0
        for ind_, data in enumerate(tqdm(val_dataloader, desc=f'Val epoch {epoch}')):
            batch = batch_to_device(data, cfg.device)
            with autocast():
                output = model(batch)
            for key, val in output.items():
                val_data[key] += [output[key]]
        for key, val in output.items():
            value = val_data[key]
            if isinstance(value[0], list):
                val_data[key] = [item for sublist in value for item in sublist]
            else:
                if len(value[0].shape) == 0:
                    val_data[key] = torch.stack(value)
                else:
                    val_data[key] = torch.cat(value, dim=0)
        loss_names = [key for key in output if 'loss' in key]
        loss_names += [key for key in output if 'score' in key]

        val_df = val_dataloader.dataset.df
        pp_out = post_process_pipeline(cfg, val_data, val_df)

        val_score = calc_metric(cfg, pp_out, val_df, "val")
        if type(val_score)!=dict:
            val_score = {f'score':val_score}

        for k, v in val_score.items():
            print(f"val_{k}: {v:.3f}")
            if neptune_run:
                neptune_run[f"val/{k}"].log(v, step=cfg.curr_step)
    
    if not os.path.exists('weights'): os.makedirs('weights')
    torch.save({"model": model.state_dict()}, f"weights/checkpoint_{neptune_run._sys_id}.pth")
