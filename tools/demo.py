#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Single-view test of action anticipation."""

import numpy as np
import os
import pickle
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter, EPICTestMeter

from slowfast.utils.metrics import MT5R
from slowfast.utils.writejson import writetestjson
from slowfast.utils.savescores import savescores

import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor


from fvcore.nn import FlopCountAnalysis

from tqdm import tqdm

logger = logging.get_logger(__name__)

from collections import defaultdict
import json

import matplotlib.pyplot as plt

def show_img(img):
    img = np.asarray(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('img_resized.jpg')

def show_img2(img1, img2, alpha=0.8, savedir=None, savename=None):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    plt.figure(figsize=(10, 10))
    plt.imshow(img1)
    plt.imshow(img2, alpha=alpha)
    plt.axis('off')
    savedir = os.path.join('att_maps','mf', savedir)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    # print(os.path.join(savedir, savename+'.png'))
    plt.savefig(os.path.join(savedir, savename+'.png'), bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close()

def my_forward_wrapper(attn_obj):
    def my_forward(x):
        (B, N, C) = x.shape
        
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_map = attn
        cls_attn_map = attn[:, :, 0, 1:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x, attn_map, cls_attn_map
    return my_forward

def get_verb_noun_names(actions_df, verb_id, noun_id):
    """
    actions_df : Dataframe mapping verb and noun id to class names
    """
    action = actions_df.loc[(actions_df['verb'] == verb_id) & (actions_df['noun'] == noun_id)]['action'].values.item()
    return action.split()[0], action.split()[1]

@torch.no_grad()
def demo(cfg):
    """
    Perform single-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    
    # Build the video model and print model statistics.
    model = build_model(cfg)
    
    cur_epoch = cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "val")
    # Enable eval mode.
    model.eval()
    import pandas as pd
    actions_df = pd.read_csv(os.path.join(cfg.EPICKITCHENS.ANNOTATIONS_DIR, cfg.EPICKITCHENS.ACTIONS_LIST))
    
    for cur_iter, (inputs, labels, video_idx, meta, future_inputs, display_inputs, label_names) in tqdm(enumerate(test_loader)):
        # Send the input videos to display for the user. 
        # Get choice of user - 0 to batch_size - 1
        # if choice >= batch_size, fetch another batch 
        # choice = get_choice(display_inputs)
        # inputs size - 4, 3, 16, 224,224
        
        choice = 0
        
        if isinstance(inputs, list):
            inputs = inputs[0]
        inputs = inputs[choice].unsqueeze(0)
        future_inputs = future_inputs[choice].unsqueeze(0)
        display_inputs = display_inputs[choice].unsqueeze(0)
        video_idx = video_idx[choice]
        labels['verb'] = labels['verb'][choice]
        labels['noun'] = labels['noun'][choice]
        labels['action'] = labels['action'][choice]
        meta['narration_id'] = meta['narration_id'][choice]
        bboxes = {}
        bboxes['hand'] = meta['orvit_bboxes']['hand'][choice].unsqueeze(0)
        bboxes['obj'] = meta['orvit_bboxes']['obj'][choice].unsqueeze(0)
        meta['orvit_bboxes'] = bboxes

        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            inputs, labels, video_idx, meta = misc.iter_to_cuda([inputs, labels, video_idx, meta])
        
            
        preds, extra_preds = model(inputs, meta)
        metadata = meta
        verb_pred = torch.argmax(extra_preds['verb']).detach().cpu().item()
        noun_pred = torch.argmax(extra_preds['noun']).detach().cpu().item()
        verb_conf = str(extra_preds['verb'][:,verb_pred].detach().cpu().item()*100)
        noun_conf = str(extra_preds['noun'][:,noun_pred].detach().cpu().item()*100)

        verb_pred, noun_pred = get_verb_noun_names(actions_df, verb_pred, noun_pred)

        verb_label = label_names['verb'][choice]
        noun_label = label_names['noun'][choice]
        
           
        # display the output
        # display(display_inputs, future_inputs, verb_label, noun_label, verb_pred, noun_pred, verb_conf, noun_conf)


import os
import sys
sys.path = [x for x  in sys.path if not (os.path.isdir(x) and 'slowfast' in os.listdir(x))]
sys.path.append(os.getcwd())

import slowfast
assert slowfast.__file__.startswith(os.getcwd()), f"sys.path: {sys.path}, slowfast.__file__: {slowfast.__file__}"

"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    if cfg.CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES

    demo(cfg)

if __name__ == "__main__":
    main()