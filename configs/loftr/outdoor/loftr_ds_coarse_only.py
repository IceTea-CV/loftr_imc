import sys
import os

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
loftr_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(os.path.join(loftr_root,"loftr_src", "config"))


from default import _CN as cfg

cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
cfg.LOFTR.MATCH_COARSE.THR = 0.2
cfg.TRAINER.CANONICAL_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]
cfg.LOFTR.FINE.ENABLE = False

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
