import torch
import os

def savescores(score_dict, cfg):
    """
    Args:
        score_dict: Contains the prediciton scores of verbs and nouns along with the action labels
    Returns:
        Save to HDF5 file
    """
    if cfg.TEST.DATASET == 'epickitchens':
        SAVEPATH = os.path.join(cfg.OUTPUT_DIR, cfg.EPICKITCHENS.TEST_LIST.strip('.pkl')+'.pt')
    torch.save(score_dict, SAVEPATH)
       
    return None