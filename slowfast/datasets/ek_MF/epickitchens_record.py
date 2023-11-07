#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

try:
    from epic_kitchens.hoa.io import save_detections
    from epic_kitchens.hoa.types import HandSide
except Exception as e:
    print(e)
from .video_record import VideoRecord
from datetime import timedelta
import time
import os
import pandas as pd
# video record

def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, '%H:%M:%S.%f')
    sec = float(timedelta(hours=x.tm_hour,
                          minutes=x.tm_min,
                          seconds=x.tm_sec).total_seconds()) + float(
        timestamp.split('.')[-1]) / 100
    return sec


class EpicKitchensVideoRecord(VideoRecord):
    def __init__(self, tup, actions_csv):
        self._index = str(tup[0])
        self._series = tup[1]
        self.actions_csv = actions_csv

    @property
    def participant(self):
        return self._series['participant_id']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_frame(self):
        return int(round(timestamp_to_sec(self._series['start_timestamp']) * self.fps))

    @property
    def end_frame(self):
        return int(round(timestamp_to_sec(self._series['stop_timestamp']) * self.fps))

    @property
    def fps(self):
        is_100 = len(self.untrimmed_video_name.split('_')[1]) == 3
        return 50 if is_100 else 60

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame

    @property
    def label(self):
        if 'verb_class' in self._series and 'noun_class' in self._series:
            action = self.actions_csv.loc[(self.actions_csv['verb'] == self._series['verb_class']) 
                    & (self.actions_csv['noun'] == self._series['noun_class'])]['id'].item()
        else:
            action = -1
        return {'verb': self._series['verb_class'] if 'verb_class' in self._series else -1,
                'noun': self._series['noun_class'] if 'noun_class' in self._series else -1,
                'action': action}
    @property
    def label_names(self):
        return{'verb': self._series['verb'],
               'noun': self._series['noun']}
       
    @property
    def metadata(self):
        return {'narration_id': self._index}

#
import pickle
import random
import slowfast.utils.distributed as du

def sample_portion_from_data(cfg, size, _video_records, _spatial_temporal_idx):
    assert len(_video_records) == len(_spatial_temporal_idx), f"len(_video_records) , len(_spatial_temporal_idx): {len(_video_records)}, {len(_spatial_temporal_idx)}"
    assert size > 0 and size <= 1, f"size: {size}"
    base = os.path.join('run_files', 'EK_data_portions')
    os.makedirs(base, exist_ok=True, mode=0o777)
    n = len(_video_records)
    n_sample = int(size * n)
    name = f'{cfg.TRAIN.DATASET}_{n_sample}_out_of_{n}.pkl'
    path = os.path.join(base, name)
    if not os.path.isfile(path) and du.is_master_proc():
        indices = random.sample(range(n), n_sample)
        with open(path, 'wb') as f:
            pickle.dump(indices,f)
    du.synchronize()
    with open(path, 'rb') as f:
        indices = pickle.load(f)
    _video_records, _spatial_temporal_idx = map(lambda x: [x[i] for i in indices], [_video_records, _spatial_temporal_idx])
    return _video_records, _spatial_temporal_idx


# bbox
import pickle
from pathlib import Path
from typing import Iterator, List, Union
try:
    from epic_kitchens.hoa import load_detections
    from epic_kitchens.hoa.types import HandSide
    RIGHT , LEFT = HandSide.RIGHT, HandSide.LEFT
except Exception as e:
    RIGHT, LEFT = None, None
    print(e)
import numpy as np
from slowfast.utils.LinkBoxes import sort_boxes
import h5py
import json
from slowfast.utils.LinkBoxes.sort_boxes import sort_boxes_sorted
from slowfast.utils.box_ops import box_xyxy_to_cxcywh, zero_empty_boxes

class EKBoxes:
    def __init__(self, cfg, boxes=None):
        from slowfast.utils.LinkBoxes.epickitchens import get_ek_boxes
        
        self.cfg = cfg
        self.cache = {} # {vid --> bbox_object}
        self.boxes_root = self.cfg.EPICKITCHENS.VISUAL_DATA_DIR
        #self.O = self.cfg.ORVIT.O
        self.O = self.cfg.HOIVIT.O
        self.U = self.cfg.HOIVIT.U
        self.T = self.cfg.DATA.NUM_FRAMES
        self.lengths = {}
        self.h5 = True
        if boxes is None:
            self.boxes = get_ek_boxes(self.boxes_root, verbose=True)
        else:
            self.boxes = boxes
        if isinstance(self.boxes, list):
            self.hand_boxes, self.boxes = self.boxes
    
    def get_boxes(self, vid, seq, nid=None):
        """
        Args:s
            vid (str): P01_01
            seq (List[int]): 1-based
        """
        if isinstance(self.boxes ,str):
            self.boxes = h5py.File(self.boxes, 'r')
        if hasattr(self, 'hand_boxes') and isinstance(self.hand_boxes ,str):
            print(self.hand_boxes)
            self.hand_boxes = h5py.File(self.hand_boxes, 'r')
        
        # creating hand and objects separately
        boxes = {'hand': [], 'obj': []}
        for i in seq:
            frametuple = self.boxes[vid].get(str(i))
            if frametuple is None:
                boxes['hand'].append(np.empty([0,5]))
                boxes['obj'].append(np.empty([0,5]))
            else:
                boxes['hand'].append(self.boxes[vid].get(str(i)).get('hand'))
                boxes['obj'].append(self.boxes[vid].get(str(i)).get('obj'))

        if hasattr(self, 'hand_boxes'):
            hand_boxes = [self.hand_boxes[vid].get(str(i), np.empty([0,5])) for i in seq]
            hand_boxes = [h[np.arange(len(h))[h[:, -1] < 2]] for h in hand_boxes] # filter objects
            # extend boxes
            boxes = [np.concatenate([h, b], axis=0) for h, b in zip(hand_boxes, boxes)]
        
        # process hand and object boxes separately
        boxes['obj'] = sort_boxes_sorted(boxes['obj'], O = self.O, saved_indices=[0,1]) # np.array [O, T, 4]
        boxes['obj'].astype(np.float32)
        boxes['hand'] = sort_boxes_sorted(boxes['hand'], O = self.U, saved_indices=[0,1]) # np.array [O, T, 4]
        boxes['hand'].astype(np.float32)
        return boxes
    
    def prepare_boxes(self, boxes, nid):
        boxes[boxes < 0] = 0
        boxes[boxes > 1] = 1
        boxes = boxes.permute(1,0,2) # T, O, 4
        boxes = box_xyxy_to_cxcywh(boxes) # T, O, 4
        boxes = zero_empty_boxes(boxes, mode='cxcywh', eps = 0.05)
        return boxes