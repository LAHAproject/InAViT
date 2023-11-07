#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import torch
from .. import utils as utils
from ..decoder import get_start_end_idx

import decord
decord.bridge.set_bridge('torch')
from decord import VideoReader

def temporal_sampling(
    num_frames, start_idx, end_idx, num_samples, start_frame=0
):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        num_frames (int): number of frames of the trimmed action clip
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
        start_frame (int): starting frame of the action clip in the untrimmed video
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, num_frames - 1).long()
    return start_frame + index


def pack_frames_to_video_clip(
    cfg, video_record, temporal_sample_index, target_fps=30, ret_seq = False,
):
    # Load video by loading its extracted frames
    path_to_video = '{}/frames/{}'.format(
        cfg.EGTEA.VISUAL_DATA_DIR,
        video_record.untrimmed_video_name
    
    )
    img_tmpl = "{:010d}.jpg"
    fps = video_record.fps
    sampling_rate = cfg.DATA.SAMPLING_RATE
    num_samples = cfg.DATA.NUM_FRAMES
    start_idx, end_idx = get_start_end_idx(
        video_record.num_frames,
        num_samples * sampling_rate * fps / target_fps,
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
    )
    # if cfg.TRAIN.DATASET == "egtea" and cfg.EGTEA.ANTICIPATION:
    #     end_idx = start_idx - fps / target_fps
<<<<<<< HEAD
    #     start_idx = end_idx - 0.5*(num_samples * sampling_rate * fps / target_fps) # Egtea anticipation period is 0.5 s
    #     if start_idx < 0:
    #         start_idx = 0
    start_idx, end_idx = start_idx + 1, end_idx + 1
=======
    #     start_idx = end_idx - cfg.EGTEA.ANT_GAP *(num_samples * sampling_rate * fps / target_fps) # Egtea anticipation period is 0.5 s by default
    #     if start_idx < 0:
    #         start_idx = 0
    start_idx, end_idx = start_idx + 1, end_idx + 1   
>>>>>>> e0ef9a0442f6ba31ffe45ac06f6b3bf13782c7de
    frame_idx = temporal_sampling(
        video_record.num_frames,
        start_idx, end_idx, num_samples,
        start_frame=video_record.start_frame
    )
    img_paths = [
        os.path.join(
            path_to_video, 
            img_tmpl.format(idx.item()
        )) for idx in frame_idx]
    frames = utils.retry_load_images(img_paths, cfg)

    if ret_seq:
        return frames, frame_idx
    return frames

def get_video_clip(
    cfg, video_record, temporal_sample_index, target_fps=60, ret_seq = False,
):
    # Load video by loading its extracted frames
    path_to_video = '{}/{}/videos/{}'.format(
        cfg.EPICKITCHENS.VISUAL_DATA_DIR,
        video_record.participant,
        video_record.untrimmed_video_name
    )
    fps = video_record.fps
    sampling_rate = cfg.DATA.SAMPLING_RATE
    num_samples = cfg.DATA.NUM_FRAMES
    start_idx, end_idx = get_start_end_idx(
        video_record.num_frames,
        num_samples * sampling_rate * fps / target_fps,
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
    )
    start_idx, end_idx = start_idx + 1, end_idx + 1
    frame_idx = temporal_sampling(
        video_record.num_frames,
        start_idx, end_idx, num_samples,
        start_frame=video_record.start_frame
    )
    #print(frame_idx)
    # print(path_to_video)
    vr = VideoReader(path_to_video+'.avi', width=456, height=256)
    frames = vr.get_batch(frame_idx.tolist())
    r_frames = frames[:,:,:,0]
    g_frames = frames[:,:,:,1]
    b_frames = frames[:,:,:,2]
    frames = torch.stack((b_frames, g_frames, r_frames))
    frames = frames.permute(1,2,3,0)
    if ret_seq:
        return frames, frame_idx
    return frames