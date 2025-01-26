# seed, checkpoints, transforms, normalize
import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
import itertools
import matplotlib.colors as mcolors
from sklearn.metrics import roc_auc_score, confusion_matrix

def my_seed_everywhere(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def seed_worker(worker_id):
    """Ensure reproducibility in DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_checkpoint(model, optimizer, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    filename = f'{path}.pth'
    torch.save(checkpoint, filename)

def get_segop_ratios(seg_op):
    ratio_map = { ## 학습을 효율화 할 수 있게 weight을 조정해본 것 --> 없애도됨. 단지 나중에 이해하고 없애기
        'seg_fast': [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
        'seg_slow': [[5, 5], [5, 5], [5, 5], [3, 7], [3, 7], [3, 7], [3, 7]],
        'consist_0': [[5, 5, 0], [5, 5, 0], [5, 5, 0], [5, 0, 5], [5, 0, 5], [5, 0, 5], [5, 0, 5]],
        'consist_1': [[1, 9, 5], [2, 8, 5], [35, 65, 50], [5, 5, 5], [65, 35, 50], [8, 2, 5], [9, 1, 5]],
        'consist': [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
        'seg_stop_fast_0': [[5, 5], [5, 5], [5, 5], [7, 3], [7, 3], [7, 3], [7, 3]],
        'seg_stop_fast_1': [[5, 5], [5, 5], [5, 5], [8, 2], [8, 2], [8, 2], [8, 2]],
        'seg_stop_fast_2': [[5, 5], [5, 5], [5, 5], [9, 1], [9, 1], [9, 1], [9, 1]],
        'non': [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
    }
    ratio = ratio_map.get(seg_op, None)

    return ratio

