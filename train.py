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
import ssl

# Import from Directory Architecture
from config import get_args_train
import utils
from dataset.train_dataset import CustomDataset_Train
import model
import losses
import optim

def get_weights_for_epoch(current_epoch, change_epoch, ratio):
    """
    Return the set of (cls_weight, seg_weight, [consist_weight]) for the given epoch.
    """
    for idx, chk_epoch in enumerate(change_epoch):
        if current_epoch < chk_epoch:
            # use ratio[idx-1] if idx>0, else ratio[0]
            if idx == 0:
                return np.array(ratio[0]) / np.sum(ratio[0])
            return np.array(ratio[idx-1]) / np.sum(ratio[idx-1])
    # If epoch beyond all transitions
    return np.array(ratio[-1]) / np.sum(ratio[-1])

def train_one_epoch(model, criterion, loader, optimizer, device, consistency_on):
    model.train()
    total_loss, total_seg_loss, total_cls_loss = 0.0, 0.0, 0.0
    total_samples = 0

    for batch in loader:
        inputs = batch['image'].to(device)
        masks  = batch['mask'].to(device)
        labels = batch['label'].unsqueeze(1).to(device)

        seg_pred, cls_pred = model(inputs)

        # filter out "mask ==3" if any
        no_mask_inds = (masks == 3).all(dim=1).all(dim=1).all(dim=1)
        valid_inds   = ~no_mask_inds
        seg_pred_valid = seg_pred[valid_inds] if valid_inds.any() else None
        masks_valid    = masks[valid_inds]    if valid_inds.any() else None

        loss, detail = criterion(
            cls_pred=cls_pred,
            seg_pred=seg_pred_valid,
            cls_gt=labels,
            seg_gt=masks_valid,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        total_loss     += loss.item() * bs
        total_seg_loss += detail['SEG_Loss'] * bs
        total_cls_loss += detail['CLS_Loss'] * bs
        total_samples  += bs

    epoch_loss = total_loss / total_samples
    seg_loss   = total_seg_loss / total_samples
    cls_loss   = total_cls_loss / total_samples

    return epoch_loss, seg_loss, cls_loss

def validate_one_epoch(model, criterion, loader, device, consistency_on):
    model.eval()
    total_loss, total_seg_loss, total_cls_loss, total_samples = 0.0,0.0,0.0,0

    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(device)
            masks  = batch['mask'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)

            seg_pred, cls_pred = model(inputs)

            no_mask_inds = (masks == 3).all(dim=1).all(dim=1).all(dim=1)
            valid_inds   = ~no_mask_inds
            seg_pred_valid = seg_pred[valid_inds] if valid_inds.any() else None
            masks_valid    = masks[valid_inds]    if valid_inds.any() else None

            loss, detail = criterion(
                cls_pred=cls_pred,
                seg_pred=seg_pred_valid,
                cls_gt=labels,
                seg_gt=masks_valid,
                consist=consistency_on
            )

            bs = inputs.size(0)
            total_loss     += loss.item() * bs
            total_seg_loss += detail['SEG_Loss'] * bs
            total_cls_loss += detail['CLS_Loss'] * bs
            total_samples  += bs

    epoch_loss = total_loss / total_samples
    seg_loss   = total_seg_loss / total_samples
    cls_loss   = total_cls_loss / total_samples
    return epoch_loss, seg_loss, cls_loss

def main():
    # Parse arguments
    args = get_args_train()

    # path_=args.path
    # layers=args.layers
    # gpu=args.gpu
    # optim=args.optim
    # EPOCHS = args.epoch
    # ver = args.ver ## version
        # st = args.st ## delete
        # de = args.de ## delete
    # clipLimit_=args.clahe_cliplimit # clahe_cliplimit 기법의 limit (사진을 limit을 주면서 자르기 때문에 cliplimit으로 함)
    # train_batch=args.batch
    # min_side_=args.size
        # lr_=args.lr_ ## lr_type
        # lr__=args.lr__ ## lr_startstep
        # lr_p=args.lr___ ## lr_patience
    # seg_op= args.seg_op
    # seg_weight_= args.seg_weight
    # feature=args.feature
    # infer=args.infer
    # external=args.external
    # weight_=args.weight

    # half= args.half
        # thr_t_f=args.thr_t_f ## model_threshold_truefalse
        # thr=args.thr ## model_threshold
    # clip_min=args.clip_min
    # clip_max=args.clip_max
    # rotate_angle = args.rotate_angle
        # rotate_p = args.rotate_p ## rotate_percentage
        # rbc_b=args.rbc_brightness
        # rbc_c=args.rbc_contrast
        # rbc_p=args.rbc_percentage
        # ela_t_f=args.ela_t_f ## elastic_truefalse
        # ela_alpha=args.ela_alpha ## elastic_alpha
        # ela_sigma=args.ela_sigma ## elastic_sigma
        # ela_alpha_aff=args.ela_alpha_aff ## elastic_alpha_affine
        # ela_p=args.ela_p ## percentage
        # gaus_t_f=args.gaus_t_f ## gaus_truefalse
    # gaus_min=args.gaus_min
    # gaus_max=args.gaus_max
        # gaus_p=args.gaus_p ## gaus_percentage
    # cordrop_t_f=args.cordrop_t_f
        # Horizontal_t_f=args.Horizontal_t_f
        # Horizontal_p = args.Horizontal_p
    # gamma_min=args.gamma_min
    # gamma_max=args.gamma_max
        # gamma_p=args.gamma_p
        # gamma_t_f=args.gamma_t_f
    # sizecrop_min_r=args.sizecrop_min_r
    # sizecrop_p=args.sizecrop_p
    # sizecrop_t_f=args.sizecrop_t_f
    # resizecrop_p=args.resizecrop_p
    # resizecrop_t_f=args.resizecrop_t_f
    # epoch_loss_ =args.epoch_loss
    # k_size_=args.k_size
    # loss_type_=args.loss_type
        # clahe_l = args.clahe_limit ## clahe_limit
    # seed_=args.seed
    
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare output directory
    current_time = datetime.datetime.now().strftime("%m%d")
    
    change_epoch = [0, 100, 120, 135, 160, 170, 175]
    
    ratio = utils.get_segop_ratios(args.seg_op)
    
    if len(ratio[0]) == 3:
        consist_ = True
    else:
        consist_ = False
    
    # 2. Seed everything for reproducibility
    model.my_seed_everywhere(seed=42)
    
    aux_params=dict(
    pooling='avg',
    dropout=0.5,
    activation=None,
    classes=1,)

    model = MultiTaskModel(layers=layers, aux_params=aux_params)
    
    
    
    
    
    
    
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
    df = pd.read_csv(args.path)
    df['img_dcm'] = df['img_dcm'].apply(lambda x: x.replace('/home/brody9512', '') if x.startswith('/home/brody9512') else x)
    df['mask_tiff'] = df['mask_tiff'].apply(lambda x: x.replace('/home/brody9512', '') if x.startswith('/home/brody9512') else x)
    


if __name__ == "__main__":
    main()