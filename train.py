# Argparser in the beginning takes up a lot of code space; what if we just use args values as is (for ex: using path as args.path)? This would save a lot of space + memory
import datetime
import os
import monai.data
import torch
import pandas as pd
import numpy as np
import monai
from monai.transforms import *
from monai.data import Dataset
import monai
from torch.utils.data import DataLoader
import ssl
import math
import shutil
from torch.optim import lr_scheduler
import torch.utils.data.dataloader
import torch.nn as nn
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score

# Import from Directory Architecture
from config import get_args_train
import utils
from dataset.train_dataset import CustomDataset_Train
from model import MultiTaskModel
import losses
import optim

def get_weights_for_epoch(current_epoch, change_epoch, ratio):
    for idx, check_epoch in enumerate(change_epoch):
        if current_epoch < check_epoch:
            return np.array(ratio[idx-1]) / np.sum(ratio[idx-1])
        
    # If current_epoch is greater than all values in change_epoch
    return np.array(ratio[-1]) / np.sum(ratio[-1])

def train_one_epoch(model, criterion, data_loader, optimizer, device):
    """
    Trains a multi-task model for one epoch.
    
    Args:
        model (nn.Module): Multi-task model (seg + cls).
        criterion (callable): Loss function with signature 
                              (cls_pred, seg_pred, cls_gt, seg_gt, consist).
        data_loader (DataLoader): Data loader for training.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        device (torch.device): 'cuda' or 'cpu'.
        ####!no longer using!#### consist_ (bool, optional): Indicates whether to include a consistency loss. 
                                   Defaults to False.
    Returns:
        dict: Dictionary containing averaged losses (epoch_loss, epoch_seg_loss, epoch_class_loss).
    """
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_class_loss = 0.0
    
    # Loop through the dataset
    for i, batch in enumerate(data_loader):
        inputs, masks, labels = batch['image'], batch['mask'], batch['label']

        # Expand dimensions for classification/segmentation shape requirements
        labels = labels.unsqueeze(1)  # e.g., (N,) -> (N,1)
        masks = masks.unsqueeze(1)    # e.g., (N,H,W) -> (N,1,H,W)

        # Move data to GPU/CPU
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(True):
            # Forward pass
            seg_pred, cls_pred = model(inputs)

            # Filter out masks that are entirely '3' if they exist
            three_mask_indices = (masks == 3).all(dim=1).all(dim=1).all(dim=1)
            valid_indices = ~three_mask_indices
            if three_mask_indices.any():
                filtered_seg_pred = seg_pred[valid_indices]
                filtered_seg_gt = masks[valid_indices]
            else:
                filtered_seg_pred = seg_pred
                filtered_seg_gt = masks

            # Compute loss
            loss, loss_detail = criterion(
                cls_pred=cls_pred,
                seg_pred=filtered_seg_pred,
                cls_gt=labels,
                seg_gt=filtered_seg_gt,
            )
            loss_value = loss.item()

            # Check for NaN/infinite loss
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training.")
                return

            # Backpropagation & optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate batch-wise losses
            batch_size = inputs.size(0)
            running_loss += loss_value * batch_size
            running_seg_loss += loss_detail['SEG_Loss'] * batch_size
            running_class_loss += loss_detail['CLS_Loss'] * batch_size

    # Compute average losses over the entire dataset
    dataset_size = len(data_loader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_seg_loss = running_seg_loss / dataset_size
    epoch_class_loss = running_class_loss / dataset_size

    # Print and return results
    print(f"Train:\n Loss: {epoch_loss:.4f} | Seg_Loss: {epoch_seg_loss:.4f} | Class_Loss: {epoch_class_loss:.4f}\n")

    return {
        'epoch_loss': epoch_loss,
        'epoch_seg_loss': epoch_seg_loss,
        'epoch_class_loss': epoch_class_loss
    }

def validate_one_epoch(model, criterion, data_loader, device):
    model.eval()
    
    # Collections for classification labels/predictions
    all_labels = []
    all_preds = []

    # Running accumulators for losses
    running_loss = 0.0
    running_seg_loss = 0.0
    running_class_loss = 0.0

    # Metrics
    confuse_metric = ConfusionMatrixMetric()
    dice_metric = DiceMetric()
    
    for i, batch in enumerate(data_loader):
        # 1) Unpack batch
        inputs = batch['image']
        masks = batch['mask']
        labels = batch['label']
        # dcm_name = batch['dcm_name']  # if you need the DICOM name later

        # 2) Adjust tensor shapes
        labels = labels.unsqueeze(1)  # (B,) -> (B, 1)
        masks = masks.unsqueeze(1)    # (B, H, W) -> (B, 1, H, W)

        # 3) Move data to device
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        # 4) Forward pass (no grad for inference)
        with torch.no_grad():
            seg_pred, cls_pred = model(inputs)

            # Filter out masks == 3 if present
            three_mask_indices = (masks == 3).all(dim=1).all(dim=1).all(dim=1)
            valid_indices = ~three_mask_indices

            if three_mask_indices.any():
                filtered_seg_pred = seg_pred[valid_indices]
                filtered_seg_gt = masks[valid_indices]
            else:
                filtered_seg_pred = seg_pred
                filtered_seg_gt = masks

            # 5) Compute loss
            loss, loss_detail = criterion(
                cls_pred=cls_pred,
                seg_pred=filtered_seg_pred,
                cls_gt=labels,
                seg_gt=filtered_seg_gt
            )
            loss_value = loss.item()

            # 6) Check for finite loss
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping evaluation.")
                return

            # 7) Accumulate losses
            batch_size = inputs.size(0)
            running_loss += loss_value * batch_size
            running_seg_loss += loss_detail['SEG_Loss'] * batch_size
            running_class_loss += loss_detail['CLS_Loss'] * batch_size

            # 8) Post-processing for classification & segmentation
            cls_pred = torch.sigmoid(cls_pred)
            seg_pred = torch.sigmoid(seg_pred)

        # 9) Collect classification labels & predictions for metrics
        all_labels.append(labels.cpu().numpy())
        all_preds.append(cls_pred.round().cpu().numpy())  # classification threshold=0.5 => round

        # 10) Update classification/segmentation metrics
        confuse_metric(y_pred=cls_pred.round(), y=labels)
        dice_metric(y_pred=seg_pred.round(), y=masks)

    # --- After loop: finalize metrics ---
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    auc_val = roc_auc_score(all_labels, all_preds)
    f1_val = f1_score(all_labels, all_preds.round())
    acc_val = accuracy_score(all_labels, all_preds.round())
    sen_val = recall_score(all_labels, all_preds.round())  # sensitivity = recall
    spe_val = precision_score(all_labels, all_preds.round(), zero_division=1)

    dice_val = dice_metric.aggregate().item()

    # Reset for next usage
    confuse_metric.reset()
    dice_metric.reset()

    # Calculate losses per sample
    dataset_size = len(data_loader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_seg_loss = running_seg_loss / dataset_size
    epoch_class_loss = running_class_loss / dataset_size

    # Prepare the output for losses
    sample_loss = {
        'epoch_loss': epoch_loss,
        'epoch_seg_loss': epoch_seg_loss,
        'epoch_class_loss': epoch_class_loss
    }
    print(
        f"Val:\n"
        f" Loss: {epoch_loss:.4f} "
        f"Seg_Loss: {epoch_seg_loss:.4f} "
        f"Class_Loss: {epoch_class_loss:.4f}\n"
    )

    sample_metrics = {
        'auc': auc_val,
        'f1': f1_val,
        'acc': acc_val,
        'sen': sen_val,
        'spe': spe_val,
        'dice': dice_val
    }

    print(
        f" AUC: {auc_val:.4f} "
        f"F1: {f1_val:.4f} "
        f"Acc: {acc_val:.4f} "
        f"Sen: {sen_val:.4f} "
        f"Spe: {spe_val:.4f} "
        f"SEG_Dice: {dice_val:.4f}\n"
    )

    return sample_loss, sample_metrics


def main():
    # Parse arguments
    args = get_args_train()
    
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    change_epoch = [0, 100, 120, 135, 160, 170, 175]
    utils.my_seed_everywhere(args.seed)
    
    # Prepare output dir ---> doublecheck this part until "print(f"Results directory: {save_dir}")" $$$$$$$$$$$
    current_time = datetime.datetime.now().strftime("%m%d")
    
    if args.infer or args.external:
        run_name = f"{args.weight}_infer" if args.infer else f"{args.weight}_ex"
        save_dir = f"/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/result/{run_name}"
    else:
        # Some naming logic
        run_name = f"{current_time}_{args.layers}_{args.seg_op}_{args.lr_type}_{args.size}_b{args.batch}_{args.feature}"
        save_dir = f"/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/result/{run_name}"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results directory: {save_dir}")
    
    df = pd.read_csv(args.path)
    df['img_dcm'] = df['img_dcm'].apply(lambda x: x.replace('/home/brody9512', '') if x.startswith('/home/brody9512') else x)
    df['mask_tiff'] = df['mask_tiff'].apply(lambda x: x.replace('/home/brody9512', '') if x.startswith('/home/brody9512') else x)
    
    # Split
    if args.external:
        test_df = df
        test_dataset = CustomDataset_Train(test_df, args, training=False)
        test_loader = DataLoader(test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0,
                                  worker_init_fn=utils.seed_worker,
                                  collate_fn=torch.utils.data.dataloader.default_collate)
    else:
        # Filter out rows where 'label' is not 0 or 1
        df_filtered = df[df['label'].isin([0, 1])]
        
        # Splitting the filtered DataFrame into train, validation, and test DataFrames
        train_df = df_filtered[df_filtered['Mode_1'] == 'train']
        val_df = df_filtered[df_filtered['Mode_1'] == 'validation']
        test_df = df_filtered[df_filtered['Mode_1'] == 'test']

        train_dataset = CustomDataset_Train(train_df, args, training=True)
        val_dataset = CustomDataset_Train(val_df, args, training=False)
        test_dataset = CustomDataset_Train(test_df, args, training=False)

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch,
                                  shuffle=True,
                                  num_workers=0,
                                  worker_init_fn=utils.seed_worker,
                                  collate_fn=torch.utils.data.dataloader.default_collate) ## ??%% monai.data.utils.default_collate is depreciated?
        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                worker_init_fn=utils.seed_worker,
                                collate_fn=torch.utils.data.dataloader.default_collate)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 worker_init_fn=utils.seed_worker,
                                 collate_fn=torch.utils.data.dataloader.default_collate)
        
    # Build Model
    aux_params = dict(pooling = 'avg', dropout = 0.5, activation = None, classes = 1)
    
    mtl_model = MultiTaskModel(layers = args.layers, aux_params = aux_params)
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    if num_gpus > 1:
        mtl_model = nn.DataParallel(mtl_model)
    mtl_model = mtl_model.to(DEVICE)
    
    # Create optimizer / scheduler
    optimizer = optim.create_optimizer(args.optim, mtl_model, args.lr_startstep)
    if args.lr_type == 'reduce':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=args.lr_patience)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    
        
    
    
    
    ############################################################################
    ratio = utils.get_segop_ratios(args.seg_op)
    
    
    
    # 4. Load your data
    df = pd.read_csv(args.path)  # Contains train, val, test splits or just train/val
    train_df = df[df["Mode_1"] == "train"]
    val_df   = df[df["Mode_1"] == "validation"]
    
    ## Create your dataset class (place it here or import it)
    train_dataset = train_dataset.CustomDataset(
        data_frame=train_df, 
        training=True, 
        rotate_angle=args.rotate_angle,
        rotate_percentage=args.rotate_percentage,
        ### put more
    ) 
    
    ## Wrap them in DataLoaders (train_loader, val_loader)
    ## this is the same for train & test; could this be simplified?
    
    

# -----------------------------------------
# Entry
# -----------------------------------------
if __name__ == "__main__":
    main()