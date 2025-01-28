# test.py
import os
import shutil
import torch
import pandas as pd
import numpy as np
import ssl
import monai.data
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, roc_curve, auc, confusion_matrix, classification_report


import config
import utils
import optim
from losses import Uptask_Loss_Test
from model import MultiTaskModel
from dataset.test_dataset import CustomDataset_Test


# -----------------------------------------
# Main
# -----------------------------------------
def main():

    args = config.get_args_test()
    
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    ratio = [[5, 5], [5, 5],[5, 5],[1,9] ,[1, 9], [1,9], [1, 9]]
    change_epoch = [0, 15, 28, 40, 55, 68, 75]
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.my_seed_everywhere(args.seed)
    
    #Saving ###
    if args.external:
        run_name = f'{args.weight}_test'
    else:
        run_name = f'{args.weight}_{args.feature}'
    save_dir = f"/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/result/{run_name}"
    if os.path.exists(save_dir): 
        shutil.rmtree(save_dir) 
    os.mkdir(save_dir)
    print(f"Results directory: {save_dir}")
    
    df = pd.read_csv(args.path)
    df['img_dcm'] = df['img_dcm'].apply(lambda x: x.replace('/home/brody9512', '') if x.startswith('/home/brody9512') else x)
    
    # Split
    if args.external:
        test_df=df
    else:
        train_df = df[df['Mode_1'] == 'train']
        val_df = df[df['Mode_1'] == 'validation']
        test_df = df[df['Mode_1'] == 'test']

    if args.half:
        # Sample half of each DataFrame (assuming each has more than 1 row)
        train_df = train_df.sample(n=len(train_df) // 2, random_state=42)
        val_df = val_df.sample(n=len(val_df) // 2, random_state=42)
        test_df = test_df.sample(n=len(test_df) // 2, random_state=42)
        
    if args.external:
        test_dataset = CustomDataset_Test(test_df, args, training=False)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=monai.data.utils.default_collate, shuffle=False, num_workers=0, worker_init_fn=utils.seed_worker)
    else:
        train_dataset = CustomDataset_Test(train_df, args, training=True)
        val_dataset = CustomDataset_Test(val_df, args, training=False)
        test_dataset = CustomDataset_Test(test_df, args, training=False)

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
    
    model = MultiTaskModel(layers = args.layers, aux_params = aux_params)
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    if num_gpus > 1:
        model = nn.DataParallel(model)
    mtl_model = model.to(DEVICE)

    # Create optimizer / scheduler
    optimizer= optim.create_optimizer(args.optim, mtl_model, args.lr_startstep)
    if args.lr_type == 'reduce':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=args.lr_patience)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    
    
    #$#$##$#$#$#$#$$
    if args.infer or args.external:
        checkpoint = torch.load(f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{args.weight}.pth')
    else:
        checkpoint = torch.load(f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{run_name}_best_model.pth')
    
    mtl_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 테스트 데이터에 대한 추론 수행
    y_true, y_prob, results = utils.test_inference_test(mtl_model, Uptask_Loss_Test(), test_loader, DEVICE)

    # Flatten the list of arrays to a list of scalars
    y_true_flat = [item[0] for item in y_true]
    y_prob_flat_0 = [item[0] for item in y_prob]

    # 소수점 다섯 자리까지 반올림
    y_prob_flat = [round(num, 4) for num in y_prob_flat_0]

    with open(os.path.join(save_dir,f'results_y_true_prob.txt'), 'w', encoding='utf-8') as f:
        f.write(f'y_true: \n{y_true_flat}\n')
        f.write(f'y_prob: \n{y_prob_flat}\n')

    print(f'y_prob: \n{y_prob_flat}\n')

    y_true_np = np.array(y_true_flat)
    y_prob_np = np.array(y_prob_flat)
    np.savez(f"{save_dir}/results_y_true_prob.npz", y_true=y_true_np, y_prob=y_prob_np)


    fpr, tpr, th = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

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

    print(f'Thresold Value: {args.model_threshold}')

    y_pred_1 = []
    for prob in y_prob:
        if prob >= args.model_threshold: 
            y_pred_1.append(1)
        else:
            y_pred_1.append(0)

    A_pred = (y_prob_np >= args.model_threshold).astype(int)

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
    ci_sensitivity_A = utils.calculate_ci(sensitivity_A, np.sum(y_true_np == 1))

    # Specificity CI
    ci_specificity_A = utils.calculate_ci(specificity_A, np.sum(y_true_np == 0))

    # 결과 출력
    print(f"Model Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})")
    print(f"Model Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})")
    print(f"Model Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n")

    print("ROC curve (area = %0.4f, 95%% CI: %0.4f-%0.4f)" % (roc_auc, ci_lower, ci_upper))

    target_names = ["True Non-PP","True PP"]

    report=classification_report(y_true, y_pred_1, target_names=target_names)

    #결과 저장 + dcm acc, fscore 추가
    with open(os.path.join(save_dir,f'results.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Classification Report:\n{report}\n')
        f.write(f'Youden index:\n{args.model_threshold}\n')

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

    print(f'Classification Report:\n{report}\n')

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

    