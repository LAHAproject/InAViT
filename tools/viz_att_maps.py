#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

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

@torch.no_grad()
def perform_viz(test_loader, model, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    
    for cur_iter, (inputs, labels, video_idx, meta) in tqdm(enumerate(test_loader)):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            inputs, labels, video_idx, meta = misc.iter_to_cuda([inputs, labels, video_idx, meta])
        
        # Perform the forward pass.
        flops = FlopCountAnalysis(model, (inputs, meta))
        print('Total flops:',flops.total())
        # print(flops.by_module())
        # print(flops.by_module_and_operator())
        preds = model(inputs, meta)
        if isinstance(preds, tuple):
            preds, extra_preds = preds

        if isinstance(labels, (dict,)):
            metadata = meta
            verb_labels, video_idx = labels['verb'], video_idx
            noun_labels, video_idx = labels['noun'], video_idx
            verb_label = verb_labels[0].detach().cpu().item()
            noun_label = noun_labels[0].detach().cpu().item()
            # turn knob = 23,190
            # mix meat = 10,28
            # cut pizza = 7,91
            # dry pan = 14,5
            # remove onion = 12,16
            # remove cover = 12,89
            # crush spice = 49,85
            # take parsley = 0,169
            # close squash = 4,116
            # pour sugar = 9,102
            verb_noun = [(23,190), (10,28), (7,91), (14,5), (12,16), (12,89), (49,85), (0,169), (4,116), (9,102)]
            if  (verb_label, noun_label) in verb_noun:
                feat = model.forward_features(inputs, meta)
                
                viz_block = my_forward_wrapper(model.blocks[-1].attn)
                y, attn_map, cls_attn_map = viz_block(feat)    
                
                attn_map = attn_map.mean(dim=1).squeeze(0).detach()
                cls_weight = cls_attn_map.mean(dim=1).view(8, 14, 14).detach().cpu()

                for i in range(cls_weight.shape[0]):
                    cls_resized = F.interpolate(cls_weight[i,:,:].view(1, 1, 14, 14), (224, 224), mode='bilinear').view(224, 224, 1)
                    img_resized = inputs[0][0,:,2*i+1,:,:].permute(1, 2, 0) * 0.5 + 0.5
                    show_img2(img_resized.cpu(), cls_resized, alpha=0.6, savedir=metadata['narration_id'][0]+'_'+str(verb_label)+'_'+str(noun_label),
                        savename=str(i))
def visualize(cfg):
    """
    Perform multi-view testing on the pretrained video model.
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
    test_loader = loader.construct_loader(cfg, "test")
    
    # # Perform single-view test on the entire dataset.
    perform_viz(test_loader, model, cfg)


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

    visualize(cfg)

if __name__ == "__main__":
    main()