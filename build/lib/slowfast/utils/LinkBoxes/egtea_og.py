from .sort import Sort
import os
import numpy as np
import pickle
import slowfast.utils.logging as logging
import slowfast.utils.distributed as du

from collections import defaultdict

AVA_VALID_SECS = range(902, 1799)
logging.setup_logging(output_dir=None)
logger = logging.get_logger(__name__)
DEBUG = False


_typ = lambda : defaultdict(list)
nested_default_dict_list = lambda : defaultdict(_typ)

BASE_DIR = "data_cache/linked_boxes"

def _get_xyxy_from_bbox(bbox):
    return list(bbox.coords[0]) + list(bbox.coords[1])

def _extract_objects_xyxyscore(detection):
    out = [_get_xyxy_from_bbox(b.bbox) + [b.score] for b in detection.objects]
    return out

def _extract_hands_xyxyscore(detection, side):
    hbboxes = [d for d in detection.hands if d.side == side]
    if len(hbboxes) == 0: return [0,0,0,0,0]
    bbox = max(hbboxes, key=lambda x: x.score)
    ret = _get_xyxy_from_bbox(bbox.bbox) + [bbox.score]
    return ret

def gen_boxes_dict(ppath, cfg, verbose=False):
    """
    pbboxes (str): path to data dir
    """
    dboxes = nested_default_dict_list()
    dhands = nested_default_dict_list()
    picklepath = os.path.join(ppath, 'handobj')
    nvid = len(os.listdir(picklepath))

    # ori boxes
    _h, _w = cfg.EGTEA.FRAME_HEIGHT, cfg.EGTEA.FRAME_WIDTH

    for ivid, vid in enumerate(os.listdir(picklepath),1):
        # only process P01_01.pkl and P01_11.pkl
        # vids = ['P01_01.pkl', 'P01_11.pkl']
        # for ivid, vid in enumerate(vids):
        if verbose:
            logger.info(f"start vid: {vid}, {ivid}/{nvid}")
        pvid = os.path.join(picklepath, vid)
        if not vid.endswith('pkl') or not os.path.isfile(pvid): continue
        vid = vid[:-4]
        bboxes = pickle.load(open(pvid,'rb'))
        
        # Objects
        for fid in bboxes.keys():
            fbboxes = bboxes[fid]

            if fbboxes['obj'] is not None:
                # EGTEA boxes are not normalized
                fbboxes['obj'][..., [0,2]] = fbboxes['obj'][..., [0,2]] / _w  
                fbboxes['obj'][..., [1,3]] = fbboxes['obj'][..., [1,3]] / _h
                
                dboxes[vid][fid] = fbboxes['obj']

            else:
                dboxes[vid][fid] = np.zeros((0,5))

            if fbboxes['hand'] is not None:
                # EGTEA boxes are not normalized
                fbboxes['hand'][..., [0,2]] = fbboxes['hand'][..., [0,2]] / _w  
                fbboxes['hand'][..., [1,3]] = fbboxes['hand'][..., [1,3]] / _h

                dhands[vid][fid] = fbboxes['hand'][:2]

            else:
                dhands[vid][fid] = np.zeros((0,5))
    #if DEBUG: break
    return dboxes, dhands

def sort_boxes_dict(dboxes, clear_dups_threshold, dboxes_sorted = None, verbose=False):
    ret = {}
    n = len(dboxes)
    for i, vid in enumerate(dboxes.keys()):
        if verbose and i%10 == 0:
            logger.info(f"start vid: {vid}, {i}/{n}")
        ret[vid] = get_vid_boxes_linked(vid, dboxes, clear_dups_threshold, dboxes_sorted = dboxes_sorted)
    return ret

def isok(boxes):
    try:
        x0,y0,x1,y1 = [boxes[:, i] for i in range(4)]
        h = y1 - y0
        w = x1 - x0
    except Exception as e:
        logger.warning(f"Error: {e}")
        return False
    return np.all(h > 0) and np.all(w > 0)

def _get_dboxes_sorted_length(d):
    if d is None: return 0
    dd = d[list(d.keys())[0]]
    fboxes = dd[list(dd.keys())[0]]
    return len(fboxes)

def filter_small(b, eps = 0.01):
    x0,y0,x1,y1,_ = b
    h, w = y1 - y0, x1 - x0
    return (h > eps) and (w > eps)

def get_vid_boxes_linked(vid, dboxes, clear_dups_threshold, dboxes_sorted):
    n_sorted = _get_dboxes_sorted_length(dboxes_sorted)
    osort = Sort(clear_dups_threshold=clear_dups_threshold, trackers_count_start=n_sorted)
    dvid = dboxes[vid]
    dvid_sorted = {}
    rng = sorted(list(dboxes[vid].keys()))
    for fid in rng:
        boxes = dvid.get(fid, np.empty((0, 5)))
        boxes = [b for b in boxes if filter_small(b)]
        boxes = np.array(boxes)
        if len(boxes) == 0: 
            boxes = np.empty((0, 5))
        if not isok(boxes):
            logger.warning(f"bad boxes: {vid}, {fid}")
            boxes = np.empty((0, 5))
        
        dvid_sorted[str(fid)] = {}
        dvid_sorted[str(fid)]['obj'] = osort.update(boxes)
        if n_sorted > 0:
            add_boxes = dboxes_sorted[vid][fid] # [n, 5]
            assert len(add_boxes) == n_sorted
            add_boxes = np.concatenate([add_boxes[:,:-1],np.arange(n_sorted).reshape(n_sorted,1)], axis=1)
            # dvid_sorted[str(fid)]['obj'] = np.concatenate([add_boxes, dvid_sorted[fid]] , axis = 0)
            dvid_sorted[str(fid)]['hand'] = add_boxes
    return dvid_sorted

def get_out_path(pbboxes):
    path = "egtea"
    os.makedirs(BASE_DIR, exist_ok=True, mode=0o777)
    path = os.path.join(BASE_DIR, path)
    return path

def dict2h5(d, out_path, verbose=False):
    import h5py, pickle


    if isinstance(d, str):
        # path to pickle
        with open(d, 'rb') as f:
            d = pickle.load(f)

    dflatten = {}
    def _rec_flat_dict(dd, prefix):
        for k ,v in dd.items():
            if isinstance(v, dict):
                _rec_flat_dict(v, f'{prefix}{k}/')
            elif isinstance(v, np.ndarray):
                dflatten[prefix+str(k)] = v
            else:
                raise Exception(f"Invalid type: {type(v)}")
    if verbose:
        logger.info(f"Flattening dictionary")
    _rec_flat_dict(d, '')
    f = h5py.File(out_path, 'w')
    if verbose:
        logger.info(f"Creating H5 file from flattened dictionary")
    f.update(dflatten)
    f.close()

def get_egtea_boxes(pbboxes, cfg, verbose=False, h5=False):
    out_path = get_out_path(pbboxes)
    _exists = os.path.exists(out_path) 
    if h5:
        _exists = _exists or os.path.exists(out_path+'.h5')
    if not _exists and du.is_master_proc(): # to generate boxes h5 file comment this file and uncomment next one
    #if du.is_master_proc(): # to generate boxes only, comment after generation
        if verbose:
            logger.info(f"Generating Egtea bboxes from {pbboxes}, out path: {out_path}")
        if verbose: logger.info(f"Generating boxes dict ... ")
        dboxes, dhands = gen_boxes_dict(pbboxes, cfg, verbose=verbose)
        clear_dups_threshold = -1
        if verbose: logger.info(f"Generating sort_boxes_dict ... ")
        sorted_boxes = sort_boxes_dict(dboxes, clear_dups_threshold, dboxes_sorted = dhands, verbose=verbose)
        if verbose: logger.info(f"Saving  to {out_path} ")
        with open(out_path, 'wb') as f:
            pickle.dump(sorted_boxes, f)
        if verbose: logger.info(f"Finished!")
    else:
        sorted_boxes = None
    if h5:
        orig_out_path = out_path
        out_path = f'{out_path}.h5'
        if not os.path.exists(out_path) and du.is_master_proc(): # comment this line to overwrite existing h5 file and create new
        #if du.is_master_proc(): # to generate boxes only, comment after generation
            if verbose:
                logger.info(f"Create h5py file {out_path} ... ")
            if sorted_boxes is None: sorted_boxes = orig_out_path
            dict2h5(sorted_boxes, out_path, verbose=True)

    du.synchronize()
    if os.path.exists(out_path):
        if h5:
            ret = out_path
        else:
            if verbose:
                logger.info(f"Load Egtea bboxes from {out_path}")
            with open(out_path, 'rb') as f:
                ret = pickle.load(f)
    return ret


if __name__ == '__main__':
    pbboxes = "/data/dataset/egtea/"
    cfg = 'slowfast/config/defaults.py'
    get_egtea_boxes(pbboxes, cfg, verbose=True)
