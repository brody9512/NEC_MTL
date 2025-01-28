# Argparser in the beginning takes up a lot of code space; what if we just use args values as is (for ex: using path as args.path)? This would save a lot of space + memory
import datetime
import os
import monai.data
import torch
import pandas as pd
import numpy as np
import monai
import matplotlib.colors as mcolors
from monai.transforms import *
import monai
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import ssl
import math
import itertools
import shutil
from torch.optim import lr_scheduler
# import torch.utils.data.dataloader
import torch.nn as nn
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, auc, confusion_matrix, classification_report

# Import from Directory Architecture
from config import get_args_train
import utils
from dataset.train_dataset import CustomDataset_Train
from model import MultiTaskModel
from losses import Uptask_Loss_Train
import optim


# -----------------------------------------
# Functions for Train
# -----------------------------------------
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


# -----------------------------------------
# Main
# -----------------------------------------
def main():
    # Parse arguments
    args = get_args_train()
    
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Segop Ratio
    ratio = utils.get_segop_ratios(args.seg_op)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    change_epoch = [0, 100, 120, 135, 160, 170, 175]
    utils.my_seed_everywhere(args.seed)
    
    # Prepare output dir ---> doublecheck this part until "print(f"Results directory: {save_dir}")" $$$$$$$$$$$
    current_time = datetime.datetime.now().strftime("%m%d")
    
    if args.infer or args.external:
        run_name = f"{args.weight}_train" if args.infer else f"{args.weight}_traintest"
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
                                  collate_fn=monai.data.utils.default_collate)
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
                                  collate_fn=monai.data.utils.default_collate) ## ??%% monai.data.utils.default_collate is depreciated? // torch.utils.data.dataloader.default_collate
        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                worker_init_fn=utils.seed_worker,
                                collate_fn=monai.data.utils.default_collate)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0, 
                                 worker_init_fn=utils.seed_worker,
                                 collate_fn=monai.data.utils.default_collate)
        
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
    
    if not (args.infer or args.external):
        lrs =[]
        prev_weights =None
        best_loss = float('inf')
        best_auc = 0.0  # 초기화: 최고의 AUC 값을 저장하기 위한 변수
        
        # 각 손실 및 측정치에 대한 빈 딕셔너리 초기화
        losses = {k: [] for k in ['train_epoch_loss', 'train_epoch_seg_loss', 'train_epoch_class_loss', 'test_epoch_loss', 'test_epoch_seg_loss', 'test_epoch_class_loss']}
        metrics = {k: [] for k in ['auc', 'f1', 'acc', 'sen', 'spe', 'dice']}

        for epoch in range(args.epoch):
            print(f"Epoch {epoch+1}\n--------------------------------------------------")
        
            if not args.seg_op =='non':
                # 현재 epoch에 대한 가중치 가져오기
                weights = get_weights_for_epoch(epoch, change_epoch, ratio)
        
                # 이전 가중치와 현재 가중치가 다르면 가중치를 출력합니다.
                if prev_weights is None or not np.array_equal(prev_weights, weights):
                    print(f"Weights for Epoch {epoch + 1}: {weights}")
                    prev_weights = weights
                
                if len(weights) == 2:
                    cls_weight, seg_weight = weights
                    train_criterion = Uptask_Loss_Train(cls_weight=cls_weight, seg_weight=seg_weight,loss_type=args.loss_type)
                    test_criterion = Uptask_Loss_Train(cls_weight=cls_weight, seg_weight=seg_weight,loss_type=args.loss_type)
        
            else:
                train_criterion = Uptask_Loss_Train(cls_weight=1.0, seg_weight=1.0,loss_type=args.loss_type)
                test_criterion = Uptask_Loss_Train(cls_weight=1.0, seg_weight=1.0,loss_type=args.loss_type)
        
            train_criterion = train_criterion.to(DEVICE)
            test_criterion = test_criterion.to(DEVICE)
            
            train_sample_loss = train_one_epoch(mtl_model,train_criterion, train_loader, optimizer, DEVICE)
            test_sample_loss, test_sample_metrics = validate_one_epoch(mtl_model, test_criterion, val_loader, DEVICE)
        
            # 결과를 각각의 딕셔너리에 추가
            for key in losses.keys():
                if 'train' in key:
                    losses[key].append(train_sample_loss[key.split('train_')[1]])
                else:
                    losses[key].append(test_sample_loss[key.split('test_')[1]])
        
            for key in metrics.keys():
                metrics[key].append(test_sample_metrics[key])
        
            # 최적의 모델 저장
            if test_sample_loss[args.epoch_loss] < best_loss:
                best_loss = test_sample_loss[args.epoch_loss]
                utils.save_checkpoint(mtl_model, optimizer, f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{run_name}')
                print('Model saved! \n')
        
            scheduler.step(metrics=test_sample_loss[args.epoch_loss])  # test_class_loss에 따라 학습률 갱신
            lrs.append(optimizer.param_groups[0]["lr"])
        
        print("Done!")
        
        #LR
        plt.plot([i+1 for i in range(len(lrs))],lrs,color='g', label='Learning_Rate')
        plt.savefig(os.path.join(save_dir,f"LR.png"))
        plt.figure(figsize=(12, 27))
        
        plt.subplot(311)
        # 훈련 데이터에 대한 손실 그래프
        plt.plot(range(args.epoch), losses['train_epoch_loss'], color='darkred', label='Train Total Loss')
        # 테스트 데이터에 대한 손실 그래프
        plt.plot(range(args.epoch), losses['test_epoch_loss'], color='darkblue', label='Val Total Loss')
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.title("Total Losses", fontsize=16)
        plt.legend(loc='upper right')

        plt.subplot(312)
        # 훈련 데이터에 대한 손실 그래프
        plt.plot(range(args.epoch), losses['train_epoch_seg_loss'], color='red', label='Train Segmentation Loss')
        plt.plot(range(args.epoch), losses['train_epoch_class_loss'], color= 'salmon' , label='Train Classification Loss')
        # 테스트 데이터에 대한 손실 그래프  
        plt.plot(range(args.epoch), losses['test_epoch_seg_loss'], color='blue', label='Val Segmentation Loss')
        plt.plot(range(args.epoch), losses['test_epoch_class_loss'], color='lightblue', label='Val Classification Loss')
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.title("SEG and CLS Losses", fontsize=16)
        plt.legend(loc='upper right')
        
        
        plt.subplot(313)
        # 훈련 데이터에 대한 메트릭 그래프
        plt.plot(range(args.epoch), metrics['f1'], color='hotpink', label='F1 Score_(CLS)')
        plt.plot(range(args.epoch), metrics['dice'], color='royalblue', label='Dice Score_(SEG)')
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Score", fontsize=11)
        plt.title("F1(CLS) and Dice Score(SEG)", fontsize=16)
        plt.legend()
        
        plt.savefig(os.path.join(save_dir,f"train_val_loss.png"))
        plt.close()
        
        checkpoint = torch.load(f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{run_name}.pth')
        mtl_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if args.infer or args.external:
        checkpoint = torch.load(f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{run_name}.pth') # args.weight
    else:
        checkpoint = torch.load(f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{run_name}.pth')
    
    mtl_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 테스트 데이터에 대한 추론 수행
    y_true, y_prob, avg_cls_loss, avg_seg_loss, dice_score, results = utils.test_inference_train(mtl_model, Uptask_Loss_Train(cls_weight=1.0,seg_weight=1.0).to(DEVICE), test_loader, DEVICE)
    
    # Flatten the list of arrays to a list of scalars
    y_true_flat = [item[0] for item in y_true]
    y_prob_flat_0 = [item[0] for item in y_prob]

    # 소수점 다섯 자리까지 반올림
    y_prob_flat = [round(num, 5) for num in y_prob_flat_0]

    print(f'y_true: \n{y_true_flat}\n')
    print(f'y_prob: \n{y_prob_flat}\n')

    with open(os.path.join(save_dir,f'results_y_true_prob.txt'), 'w', encoding='utf-8') as f:
        f.write(f'y_true: \n{y_true_flat}\n')
        f.write(f'y_prob: \n{y_prob_flat}\n')

    y_true_np = np.array(y_true_flat)
    y_prob_np = np.array(y_prob_flat)
    np.savez(f"{save_dir}/results_y_true_prob.npz", y_true=y_true_np, y_prob=y_prob_np)
    
    fpr, tpr, th = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    youden = np.argmax(tpr-fpr)

    # Calculate 95% CI for AUC using bootstrap
    ci_lower, ci_upper = utils.calculate_auc_ci(np.array(y_true_flat), np.array(y_prob_flat))

    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f, 95%% CI: %0.2f-%0.2f)" % (roc_auc, ci_lower, ci_upper))
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)

    # 그림을 저장
    plt.savefig(os.path.join(save_dir,f"roc_curve.png"))
    plt.close()

    if args.model_threshold_truefalse:
        thr_val = args.model_threshold
    else:
        thr_val = th[youden]

    y_pred_1 = []
    for prob in y_prob:
        if prob >= thr_val: 
            y_pred_1.append(1)
        else:
            y_pred_1.append(0)
    
    A_pred = (y_prob_np >= thr_val).astype(int)

    # 정확도 (Accuracy) 계산
    accuracy_A = accuracy_score(y_true_np, A_pred)

    # 민감도 (Sensitivity) 계산 (Recall과 동일)
    sensitivity_A = recall_score(y_true_np, A_pred)

    # 특이도 (Specificity) 계산
    # Specificity = TN / (TN + FP)
    conf_matrix_A = confusion_matrix(y_true_np, A_pred)
    specificity_A = conf_matrix_A[0, 0] / (conf_matrix_A[0, 0] + conf_matrix_A[0, 1])

    # Accuracy CI
    ci_accuracy_A = utils.calculate_ci(accuracy_A, len(y_true_np))

    # Sensitivity CI
    ci_sensitivity_A = utils.calculate_c(sensitivity_A, np.sum(y_true_np == 1))

    # Specificity CI
    ci_specificity_A = utils.calculate_c(specificity_A, np.sum(y_true_np == 0))

    # 결과 출력
    print(f"Model Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})")
    print(f"Model Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})")
    print(f"Model Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n")

    print("ROC curve (area = %0.4f, 95%% CI: %0.4f-%0.4f)" % (roc_auc, ci_lower, ci_upper))

    target_names = ["True Non-PP","True PP"]

    report=classification_report(y_true, y_pred_1, target_names=target_names)

    #결과 저장 + dcm acc, fscore 추가
    with open(os.path.join(save_dir,f'results.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Average Classification Loss: {avg_cls_loss}\n')
        f.write(f'Average Segmentation Loss: {avg_seg_loss}\n')
        f.write(f'Dice Score: {dice_score}\n')
        f.write(f'Classification Report:\n{report}\n')
        f.write(f'Youden index:\n{thr_val}\n')

        # Model Performance Metrics 저장
        f.write(f"\nModel Performance Metrics:\n")
        f.write(f"Model Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})\n")
        f.write(f"Model Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})\n")
        f.write(f"Model Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n")
        f.write(f"ROC curve (area = {roc_auc:.2f}, 95% CI: {ci_lower:.4f}-{ci_upper:.4f})\n")

        for dcm_name, metrics in results.items():
            f.write(f'\n dcm_name: {dcm_name}\n')
            for metric, value in metrics.items():
                f.write(f'  {metric}: {value}\n')

    target_names = ["True Non-PP","True PP"]
    target_names_1=["Pred Non-PP","Pred PP"]
    
    # Confusion matrix 계산
    cm = confusion_matrix(y_true, y_pred_1)

    # Confusion matrix 그리기
    utils.plot_confusion_matrix(cm, target_names,target_names_1,save_path=os.path.join(save_dir, f"confusion_matrix.png"))#'/path/to/save/image.png
    

# -----------------------------------------
# Entry
# -----------------------------------------
if __name__ == "__main__":
    main()