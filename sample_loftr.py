import sys
import os
import numpy as np
import _testimportmultiple

loftr_path = '/mnt/nas2/ksy/IMC2025/LoFTR'
sys.path.append(loftr_path) # TODO: Consider making this relative
from src.utils.imc2025_loftr import *

def lower_config(yacs_cfg):
    from yacs.config import CfgNode as CN
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def build_LoFTR_quant():
    from src.loftr.loftr import LoFTR
    from src.config.default import get_cfg_defaults as get_cfg_defaults_loftr

    cfg = get_cfg_defaults_loftr()
    cfg.merge_from_file(os.path.join(loftr_path,"configs/loftr/outdoor/loftr_ds_coarse_only.py"))
    match_cfg = lower_config(cfg)
    match_cfg['loftr']['match_coarse']['thr'] = 0.4
    match_cfg['loftr']['coarse']['temp_bug_fix'] = False

    matcher = LoFTR(config=match_cfg['loftr'])
    # load checkpoints
    state_dict = torch.load(os.path.join(loftr_path,"weight/outdoor_ds.ckpt"), map_location="cpu")["state_dict"]
    matcher.load_state_dict(state_dict, strict=True)
    matcher.eval()

    return matcher

def build_LoFTR():
    from src.loftr.loftr import LoFTR
    from src.config.default import get_cfg_defaults as get_cfg_defaults_loftr
        
    cfg = get_cfg_defaults_loftr()
    cfg.merge_from_file(os.path.join(loftr_path,"configs/loftr/outdoor/loftr_ds.py"))
    match_cfg = lower_config(cfg)
    match_cfg['loftr']['match_coarse']['thr'] = 0.4
    match_cfg['loftr']['coarse']['temp_bug_fix'] = False
    matcher = LoFTR(config=match_cfg['loftr'])
    # load checkpoints
    state_dict = torch.load(os.path.join(loftr_path, "weight/outdoor_ds.ckpt"), map_location="cpu")["state_dict"]
    matcher.load_state_dict(state_dict, strict=True)
    matcher.eval()

    return matcher


import kornia as K
device = K.utils.get_cuda_device_if_available(0)

loftr_model = build_LoFTR().to(device)
mkpts1, mkpts2 = pairwise_loftr_match_raw(loftr_model,  '/mnt/nas2/ksy/IMC2025/train/ETs/et_et000.png', '/mnt/nas2/ksy/IMC2025/train/ETs/et_et001.png',  device=device)
print(mkpts1.shape)
#Forced to quantize 8
loftr_quant_model = build_LoFTR_quant().to(device)
mkpts1, mkpts2 = pairwise_loftr_match_quant(loftr_quant_model,  '/mnt/nas2/ksy/IMC2025/train/ETs/et_et000.png', '/mnt/nas2/ksy/IMC2025/train/ETs/et_et001.png',  device=device)
print(mkpts1.shape)
