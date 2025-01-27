### DONE for both train and test ### 
# my_seed_everywhere, seed_worker, save_checkpoint, get_segop_ratios, plot_confusion_matrix, calculate_ci, calculate_auc_ci, test_inference
import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
import itertools
import matplotlib.colors as mcolors
from monai.metrics import DiceMetric
from sklearn.metrics import roc_auc_score, accuracy_score

def my_seed_everywhere(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multiGPU
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

def plot_confusion_matrix(cm, args, true_labels,pred_labels, title='Confusion matrix', cmap=plt.cm.Blues,save_path=None):
    
    plt.figure(figsize=(6, 5))

    # Normalize the confusion matrix by rows (i.e., by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Define a colormap with normalization from 0 to 1
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Plot the confusion matrix with normalization
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap, norm=norm)
    plt.title(title) #### this is the only difference between test and train 

    # Colorbar
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)

    tick_marks = np.arange(len(true_labels))
    plt.xticks(tick_marks, pred_labels, rotation=0, fontsize=10)
    plt.yticks(tick_marks, true_labels, rotation=90, fontsize=10, ha='right')
    
    # 행렬의 각 셀에 대해 반복
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # 현재 셀의 값이 해당 행의 전체 합계에 대한 비율을 계산
        percent = cm[i, j] / cm[i, :].sum()
        
        # 퍼센트 표시 (위)
        plt.text(j, i-0.1, f"{percent:.2f}", 
                 horizontalalignment="center",
                 verticalalignment="center",
                 fontsize=20,
                 color="white" if cm_normalized[i, j] > args.model_threshold else "black")
                
        # 현재 셀에 숫자를 표시 (아래)
        plt.text(j, i+0.1, f"({format(cm[i, j], 'd')})", 
                 horizontalalignment="center",
                 verticalalignment="center",
                 fontsize=20,
                 color="white" if cm_normalized[i, j] > args.model_threshold else "black")
    
    plt.tight_layout()

    # 이미지 저장
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_ci(metric, n, z=1.96):
    # 95% 신뢰 구간(CI) 계산을 위한 함수 정의
    se = np.sqrt((metric * (1 - metric)) / n)
    ci_lower = metric - z * se
    ci_upper = metric + z * se
    
    return ci_lower, ci_upper

def calculate_auc_ci(y_true, y_probs, args, n_bootstraps=1000, alpha=0.95):
    
    bootstrapped_aucs = []
    rng = np.random.RandomState(args.seed)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = roc_auc_score(y_true[indices], y_probs[indices])
        bootstrapped_aucs.append(score)

    sorted_scores = np.array(bootstrapped_aucs)
    sorted_scores.sort()

    lower = (1.0 - alpha) / 2.0
    upper = 1.0 - lower
    lower_bound = np.percentile(sorted_scores, lower * 100)
    upper_bound = np.percentile(sorted_scores, upper * 100)
    
    return lower_bound, upper_bound

def test_inference_train(model, criterion, data_loader, device, threshold=0.5):
    model.eval()
    y_true = []
    y_prob = []
    total_cls_loss = 0.0
    total_seg_loss = 0.0
    total_samples = 0
    dice_metric_batch = DiceMetric(include_background=False, reduction='mean')  # For batch-level dice score
    dice_metric_epoch = DiceMetric(include_background=False, reduction='mean')  # For epoch-level dice score
    results = {}  # Empty dictionary to store results

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, masks, labels, dcm_name = data['image'], data['mask'], data['label'], data['dcm_name']

            labels = labels.unsqueeze(1)
            masks = masks.unsqueeze(1)

            inputs = inputs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Make predictions with the model.
            seg_pred, cls_pred = model(inputs)

            # Calculate losses
            loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, cls_gt=labels, seg_gt=masks)
            total_cls_loss += loss_detail['CLS_Loss'] * inputs.size(0)
            total_seg_loss += loss_detail['SEG_Loss'] * inputs.size(0)
            total_samples += inputs.size(0)

            # Post-processing
            cls_pred = torch.sigmoid(cls_pred)
            seg_pred = torch.sigmoid(seg_pred)

            cls_pred_bin = (cls_pred > threshold).float()
            seg_pred_bin = (seg_pred > threshold).float()

            acc = accuracy_score(labels.cpu().numpy(), cls_pred_bin.cpu().numpy())

            dice_score_batch = dice_metric_batch(y_pred=seg_pred_bin, y=masks).mean().item()
            formatted_dice_score_batch = f"{dice_score_batch:.4f}"
            results[dcm_name[0]] = {'Accuracy': acc, 'Dice Score': formatted_dice_score_batch}

            y_true.extend(labels.cpu().numpy())
            y_prob.extend(cls_pred.cpu().numpy())

            dice_metric_epoch(y_pred=seg_pred_bin, y=masks)

    average_cls_loss = total_cls_loss / total_samples
    average_seg_loss = total_seg_loss / total_samples
    dice_score_epoch = dice_metric_epoch.aggregate().item()
    dice_metric_epoch.reset()

    print('cls_loss:', average_cls_loss,'SEG_loss:',average_seg_loss, 'Dice Score:', f'{dice_score_epoch:.4f}')
    
    return y_true, y_prob, average_cls_loss, average_seg_loss, dice_score_epoch, results

def test_inference(model, criterion, data_loader, device, threshold=0.5):
    model.eval()

    y_true = []
    y_prob = []
    total_samples = 0
    results = {}  # Empty dictionary to store results
    
    print('inference start')
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs,labels, dcm_name = data['image'], data['label'], data['dcm_name']

            labels = labels.unsqueeze(1)  # labels의 크기를 (2,)에서 (2, 1)로 바꿉니다.

            inputs = inputs.to(device) 
            labels = labels.to(device)

            # Make predictions with the model.
            _, cls_pred = model(inputs)

            # 여기서 cls_pred가 튜플일 경우, 올바른 출력을 선택합니다.
            if isinstance(cls_pred, tuple):
                cls_pred = cls_pred[1]  # 예를 들어 첫 번째 요소가 분류 작업의 출력인 경우

            
            # Calculate losses
            loss= criterion(cls_pred=cls_pred, cls_gt=labels)
            total_samples += inputs.size(0)

            # Post-processing
            cls_pred = torch.sigmoid(cls_pred)

            cls_pred_bin = (cls_pred > threshold).float()

            acc = accuracy_score(labels.cpu().numpy(), cls_pred_bin.cpu().numpy())
            
            results[dcm_name[0]] = {'Accuracy': acc}

            y_true.extend(labels.cpu().numpy())
            y_prob.extend(cls_pred.cpu().numpy())
            
    print('cls_loss:', loss)
    
    return y_true, y_prob,results
