# Argparser in the beginning takes up a lot of code space; what if we just use args values as is (for ex: using path as args.path)? This would save a lot of space + memory
import datetime
import os
import torch
import pandas as pd
import numpy as np
import monai
from monai.transforms import *
from monai.data import Dataset
from torch.utils.data import DataLoader

# Import from Directory Architecture
import utils  # or any other utility you need
import model
import losses
import optim
import config


def main():
    # 1. Parse arguments
    args = config.get_args_train()

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
    change_epoch = [0, 100, 120, 135, 160, 170, 175]
    
    # Dictionary mapping seg_op values to their corresponding ratios
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

    # Retrieve the ratio based on the seg_op value
    ratio = ratio_map.get(seg_op, None)

    # Check if ratio is None, meaning an invalid seg_op was provided
    if ratio is None:
        raise ValueError(f"Invalid seg_op value: {seg_op}")
    
    if len(ratio[0]) == 3:
        consist_ = True
    else:
        consist_ = False
    
    # 2. Seed everything for reproducibility
    model.my_seed_everywhere(seed=42)
    
    # 3. Set up GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 4. Load your data
    df = pd.read_csv(args.path)  # Contains train, val, test splits or just train/val
    train_df = df[df["Mode_1"] == "train"]
    val_df   = df[df["Mode_1"] == "validation"]
    
    # Create your dataset class (place it here or import it)
    train_dataset = CustomDataset(
        data_frame=train_df, 
        training=True, 
        rotate_angle=args.rotate_angle,
        rotate_p=args.rotate_p,
        ### put more
    ) 
    
    # Wrap them in DataLoaders (train_loader, val_loader)
    # this is the same for train & test; could this be simplified?
    if not external:
        train_dataset = CustomDataset(train_df, training=True, apply_voi=False, hu_threshold=None, clipLimit=clipLimit_, min_side=min_side_)
        val_dataset = CustomDataset(val_df, training=False, apply_voi=False, hu_threshold=None, clipLimit=clipLimit_, min_side=min_side_)
        test_dataset = CustomDataset(test_df, training=False, apply_voi=False, hu_threshold=None, clipLimit=clipLimit_, min_side=min_side_)

        batch_size_train = train_batch
        batch_size_val = 1
        batch_size_test = 1

        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, collate_fn=monai.data.utils.default_collate, shuffle=True, num_workers=0, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_val, collate_fn=monai.data.utils.default_collate, shuffle=False, num_workers=0, worker_init_fn=seed_worker)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_test, collate_fn=monai.data.utils.default_collate, shuffle=False, num_workers=0, worker_init_fn=seed_worker)
    else:
        batch_size_test=1
        test_dataset = CustomDataset(test_df, training=False, apply_voi=False, hu_threshold=None, clipLimit=clipLimit_, min_side=min_side_)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_test, collate_fn=monai.data.utils.default_collate, shuffle=False, num_workers=0, worker_init_fn=seed_worker)
        
    # Build your model 
    
    # (Optional) If multiple GPUs:
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    # 6. Create optimizer (from `optim.py`)
    optimizer = optim.create_optimizer(name="adam", net=model, lr=args.lr)
    # or pass other arguments as needed, e.g.: 
    # optimizer = create_optimizer(args.optim, model, args.lr)
    
    # 7. Define your loss function
    criterion = losses.Uptask_Loss_Train()  
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