import datetime
import os
import torch
import pandas as pd
import numpy as np
import monai

from torch.utils.data import DataLoader
from utils import my_seed_everywhere  # or any other utility you need
from model import MultiTaskModel
from losses import Uptask_Loss  # or Dice_BCE_Loss, etc.
from optim import create_optimizer  # Suppose you define this in `optim.py`
from config import get_args_train


def main():
    # 1. Parse arguments
    args = get_args_train()
    path_=args.path
    layers=args.layers
    gpu=args.gpu
    optim=args.optim
    EPOCHS = args.epoch
    ver = args.ver
    st = args.st
    de = args.de
    clipLimit_=args.clahe
    train_batch=args.batch
    min_side_=args.size
    lr_=args.lr_
    lr__=args.lr__
    lr_p=args.lr___
    seg_op= args.seg_op
    seg_weight_= args.seg_weight
    feature=args.feature
    infer=args.infer
    external=args.external
    weight_=args.weight
    cbam_ = args.cbam
    half= args.half
    thr_t_f=args.thr_t_f
    thr=args.thr
    clip_min=args.clip_min
    clip_max=args.clip_max
    rotate_angle = args.rotate_angle
    rotate_p = args.rotate_p
    rbc_b=args.rbc_b
    rbc_c=args.rbc_c
    rbc_p=args.rbc_p
    ela_t_f=args.ela_t_f
    ela_alpha=args.ela_alpha
    ela_sigma=args.ela_sigma
    ela_alpha_aff=args.ela_alpha_aff
    ela_p=args.ela_p
    gaus_t_f=args.gaus_t_f
    gaus_min=args.gaus_min
    gaus_max=args.gaus_max
    gaus_p=args.gaus_p
    cordrop_t_f=args.cordrop_t_f
    Horizontal_t_f=args.Horizontal_t_f
    Horizontal_p = args.Horizontal_p
    gamma_min=args.gamma_min
    gamma_max=args.gamma_max
    gamma_p=args.gamma_p
    gamma_t_f=args.gamma_t_f
    sizecrop_min_r=args.sizecrop_min_r
    sizecrop_p=args.sizecrop_p
    sizecrop_t_f=args.sizecrop_t_f
    resizecrop_p=args.resizecrop_p
    resizecrop_t_f=args.resizecrop_t_f
    epoch_loss_ =args.epoch_loss
    k_size_=args.k_size
    loss_type_=args.loss_type
    clahe_l = args.clahe_limit
    seed_=args.seed

    current_time = datetime.datetime.now().strftime("%m%d")
    change_epoch = [0, 100, 120, 135, 160, 170, 175] ## [0, 100, 120, 130, 160, 170, 175]
    
    # 2. Seed everything for reproducibility
    my_seed_everywhere(seed=42)
    
    # 3. Set up GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 4. Load your data
    df = pd.read_csv(args.path)  # Contains train, val, test splits or just train/val
    train_df = df[df["Mode_1"] == "train"]
    val_df   = df[df["Mode_1"] == "validation"]
    
    # Create your dataset class (place it here or import it)
    
    # Wrap them in DataLoaders (train_loader, val_loader)
     
    # Build your model 
    
    # (Optional) If multiple GPUs:
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    # 6. Create optimizer (from `optim.py`)
    optimizer = create_optimizer(name="adam", net=model, lr=args.lr)
    # or pass other arguments as needed, e.g.: 
    # optimizer = create_optimizer(args.optim, model, args.lr)
    
    # 7. Define your loss function
    criterion = Uptask_Loss(...)  
    # You can pass any arguments you need for multi-task weighting, etc.

    # 8. Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epoch):
        print(f"Epoch [{epoch+1}/{args.epoch}]")

        # --- Train ---
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # --- Validate ---
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.weight}_best.pth")
            print("New best model saved!")

    print("Training complete. Best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()