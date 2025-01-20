# seed functions --> does this go directly into the model (from 경성구 연구원's github it does)?
# What goes in Util? checkpoints, saves? 
import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
import itertools
import matplotlib.colors as mcolors
from sklearn.metrics import roc_auc_score, confusion_matrix

# do these go directly in the main py file?
def my_seed_everywhere(seed: int = 42):
    random.seed(seed)  # random
    np.random.seed(seed)  # numpy
    os.environ["PYTHONHASHSEED"] = str(seed)  # os
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multiGPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Function to initialize seeds in DataLoader workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def save_checkpoint(model,optimizer,longpath):
      checkpoint = {
                  #'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
      } # save all important stuff
      filename = '{}.pth'.format(longpath)
      torch.save(checkpoint , filename) 