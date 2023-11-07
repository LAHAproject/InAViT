This repository contains the official PyTorch implementation of the paper:

# InAViT - Interaction Region Visual Transformer for Egocentric Action Anticipation
Debaditya Roy, Ramanathan Rajendiran, and Basura Fernando  
Accepted at WACV 2024 [paper](https://arxiv.org/pdf/2211.14154.pdf)

* **Ranked 1 on EK100 action anticipation for both val and test set**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/interaction-visual-transformer-for-egocentric/action-anticipation-on-epic-kitchens-100)](https://paperswithcode.com/sota/action-anticipation-on-epic-kitchens-100?p=interaction-visual-transformer-for-egocentric)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/interaction-visual-transformer-for-egocentric/action-anticipation-on-epic-kitchens-100-test)](https://paperswithcode.com/sota/action-anticipation-on-epic-kitchens-100-test?p=interaction-visual-transformer-for-egocentric)



* Best results on EGTEA, see paper
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/interaction-visual-transformer-for-egocentric/action-anticipation-on-egtea)](https://paperswithcode.com/sota/action-anticipation-on-egtea?p=interaction-visual-transformer-for-egocentric)	

If you find this code useful and use this code, please cite our paper

```
@article{roy2022interaction,
  title={Interaction Visual Transformer for Egocentric Action Anticipation},
  author={Roy, Debaditya and Rajendiran, Ramanathan and Fernando, Basura},
  journal={arXiv preprint arXiv:2211.14154},
  year={2022}
  }
```

# Installation
  First, create a conda virtual environment and activate it:
  ```
  conda create -n inavit python=3.8.5 -y
  source activate inavit
  ```
  Then, install the following packages:

  * torchvision: pip install torchvision or conda install torchvision -c pytorch
  * fvcore: pip install 'git+https://github.com/facebookresearch/fvcore'
  * simplejson: pip install simplejson
  * einops: pip install einops
  * timm: pip install timm
  * PyAV: conda install av -c conda-forge
  * psutil: pip install psutil
  * scikit-learn: pip install scikit-learn
  * OpenCV: pip install opencv-python
  * tensorboard: pip install tensorboard
  * matplotlib: pip install matplotlib
  * pandas: pip install pandas
  * ffmeg: pip install ffmpeg-python 
  * decord: pip install decord
  * filterpy: pip install filterpy
  * h5py: pip install h5py 

# Bounding Box Extraction
  * Extract hand and object box features for EK100 using https://github.com/ddshan/hand_object_detector 
  * Store as h5 file into data_cache/linked_boxes/epickitchens.h5

# Setup
  1. Change the following parameters in **slowfast/config/defaults.py**

  * Download EK100 RGB frames from the dataset provider https://data.bris.ac.uk/data/dataset/3h91syskeag572hl6tvuovwv4d 
  and update the path in 
  ```_C.EPICKITCHENS.VISUAL_DATA_DIR = "frame_path"```

  * Path to Epic-Kitchens Annotation directory
  Download the [EK100 annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations)
  and add that location to this 
  ```_C.EPICKITCHENS.ANNOTATIONS_DIR = "annotation_path"```

  * Verb-Noun to Action Mapping
  Download the (actions.csv)[https://github.com/fpv-iplab/rulstm/blob/master/RULSTM/data/ek100/actions.csv] and save it to 
  anntotion directory _C.EPICKITCHENS.ANNOTATIONS_DIR
  
  2. Download MotionFormer model for EK100 https://dl.fbaipublicfiles.com/motionformer/ek_motionformer_224_16x4.pyth and save it to CP/
  
# Training

  ``` python tools/run_net.py --cfg configs/HOIVIT/EK_INAVIT_MF_ant.yaml TRAIN.ENABLE True TEST.ENABLE False```
  * All the training parameters are in configs/HOIVIT/EK_INAVIT_MF_ant.yaml. This config is for 4x24GB Nvidia A5000 GPUs.
  * Please edit the parameters NUM_GPUS, TRAIN.BATCH_SIZE to accomodate to your GPUs.

# Testing 
  ``` python tools/run_net.py --cfg configs/HOIVIT/EK_INAVIT_MF_ant.yaml TRAIN.ENABLE False TEST.ENABLE True```

# Precursors
This code is based on the repositories of [ORVIT](https://github.com/eladb3/ORViT) and [MotionFormer](https://github.com/facebookresearch/Motionformer)


# Acknowledgment

This research/project is supported in part by the National Research Foundation, Singapore under its AI Singapore Program (AISG Award Number: AISG-RP-2019-010) and by the National Research Foundation Singapore and DSO National Laboratories under the AI Singapore Programme (AISG Award No: AISG2-RP-2020-016). This research is also supported by funding allocation to B.F. by the Agency for Science, Technology and Research (A*STAR) under its SERC Central Research Fund (CRF), as well as its Centre for Frontier AI Research (CFAR).

  
In case of issues, please write to debadityaroy5555 at gmail dot com
