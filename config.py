### Done ###
# argparse_train
# argparse_test
import argparse


def get_args_train():
    """Arguments for training + validation (multi-task)"""
    parser = argparse.ArgumentParser(description="Training Arguments")
    
    # Basic I/O and Paths
    parser.add_argument('--path', type=str, default='/workspace/changhyun/nec_ch/csv_xlxs/pneumoperiT_modified_n3861_final_20240206_pch.csv', help='/path/to/your/training_csv.csv')
    parser.add_argument('--gpu', type=str,default='2')
    parser.add_argument('--infer', action='store_true', help='Inference mode only')
    parser.add_argument('--external', action='store_true', help='Use external data only? (Testing on external set)')
    
    # Model & Training
    parser.add_argument('--optim', type=str,default='adam', help='Optimizer type.')
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--ver', type=int, default=6)
    parser.add_argument('--layers', type=str,default='densenet121', choices=['densenet121', 'densenet169','densenet201','densenet161','resnext50_32x4d','se_resnet50','se_resnet101','se_resnext50_32x4d', 'se_resnext101_32x4d','resnext101_32x8d', 'inceptionresnetv2', 'mit_b0','mit_b1','mit_b2','mit_b3','resnet101','resnet152','vgg16','vgg19','inceptionv4','mobilenet_v2','resnet50','resnet101','resnext50_32x4d','resnext101_32x4d','inceptionv4','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','vgg16','vgg19','resnext101_32x8d'])
    parser.add_argument('--batch', type=int, default=18, help='Train batch size.')
    parser.add_argument('--size', type=int, default=512, help='Target image size.')
    parser.add_argument('--lr_', type=str, default='reduce', choices=['step', 'reduce'] )
    parser.add_argument('--lr__', type=float, default=0.00005)
    parser.add_argument('--lr___', type=int, default=12)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--weight', type=str,default='ver7_densenet121_size_512_b18_sche_False_consist_False_valloss_ep120_30_60___best_model_')
    parser.add_argument('--seed', type=int, default=42)

    # Classification & Segmentation Options
    parser.add_argument('--seg_weight', action='store_true', default=False)
    parser.add_argument('--seg_op', type=str,  default='non', choices=['non', 'seg_fast','seg_slow','seg_stop_fast_2','seg_stop_fast_0','seg_stop_fast_1','consist_0','consist_1','consist'])
    parser.add_argument('--feature', type=str,default='_')
    parser.add_argument('--epoch_loss', type=str, default='epoch_loss', choices=['epoch_class_loss', 'epoch_loss'], help='Which loss to track for best model checkpoint?')

    # Data augmentations: Cropping, clipping, etc.
    parser.add_argument('--st', type=int,default=0)
    parser.add_argument('--de', type=int, default=0)
    parser.add_argument('--clahe', type=float, default=2.0)    
    parser.add_argument('--clip_min', type=float, default=0.5)
    parser.add_argument('--clip_max', type=float, default=99.5)
    parser.add_argument('--clahe_limit', type=int, default=8)    
    parser.add_argument('--rotate_angle', type=float, default=30) 
    parser.add_argument('--rotate_p', type=float, default=0.8)
    parser.add_argument('--rbc_b', type=float, default=0.2)
    parser.add_argument('--rbc_c', type=float, default=0.2)
    parser.add_argument('--rbc_p', type=float, default=0.5)
    parser.add_argument('--ela_t_f', action='store_true')
    parser.add_argument('--ela_alpha', type=float, default=30)
    parser.add_argument('--ela_sigma', type=float, default=1.5)
    parser.add_argument('--ela_alpha_aff', type=float, default=0.9)
    parser.add_argument('--ela_p', type=float, default=0.25)
    parser.add_argument('--gaus_t_f', action='store_true', default=True)
    parser.add_argument('--gaus_min', type=float, default=10.0)
    parser.add_argument('--gaus_max', type=float, default=50.0)
    parser.add_argument('--gaus_p', type=float, default=0.5)
    parser.add_argument('--cordrop_t_f', action='store_true')
    parser.add_argument('--Horizontal_p', type=float, default=0.25)
    parser.add_argument('--Horizontal_t_f', action='store_true')
    parser.add_argument('--gamma_min', type=float, default=80.0)
    parser.add_argument('--gamma_max', type=float, default=120.0)
    parser.add_argument('--gamma_p', type=float, default=0.5)
    parser.add_argument('--gamma_t_f', action='store_true')
    parser.add_argument('--sizecrop_min_r', type=float, default=0.8)
    parser.add_argument('--sizecrop_p', type=float, default=0.5)
    parser.add_argument('--sizecrop_t_f', action='store_true')
    parser.add_argument('--resizecrop_p', type=float, default=0.5)
    parser.add_argument('--resizecrop_t_f', action='store_true')
    parser.add_argument('--thr_t_f', action='store_true')
    parser.add_argument('--thr', type=float, default=0.24387007)
    
    # Multi-task weighting
    parser.add_argument('--k_size', type=int, default=8, choices=[2,4,8,16,32] )
    parser.add_argument('--loss_type', type=str, default='bc_di', choices=['bc_di','bc_iou','bc_tv','fo_di','fo_tv','fo_iou'], help='Seg+Cls loss combination.')    

    return parser.parse_args()

def get_args_test():
    """Arguments for external inference-only script"""    
    parser = argparse.ArgumentParser(description="Training Arguments")
    
    parser.add_argument('--path', type=str, default='/workspace/changhyun/nec_ch/csv_xlxs/pneumoperiT_modified_0_and_1_No_lateral_external_240306.csv')
    parser.add_argument('--gpu', type=str,default='2')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--external', action='store_true')
    
    parser.add_argument('--optim', type=str,default='adam', help='Optimizer type.')
    parser.add_argument('--epoch', type=int, default=125)
    parser.add_argument('--ver', type=int, default=6)
    parser.add_argument('--layers', type=str,default='densenet121', choices=['densenet121', 'densenet169','densenet201','densenet161','resnext50_32x4d','se_resnet50','se_resnet101','se_resnext50_32x4d', 'se_resnext101_32x4d','resnext101_32x8d', 'inceptionresnetv2', 'mit_b0','mit_b1','mit_b2','mit_b3','resnet101','resnet152','vgg16','vgg19','inceptionv4','mobilenet_v2','resnet50','resnet101','resnext50_32x4d','resnext101_32x4d','inceptionv4','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','vgg16','vgg19','resnext101_32x8d'])
    parser.add_argument('--batch', type=int, default=18, help='Train batch size.')
    parser.add_argument('--size', type=int, default=512, help='Target image size.')
    parser.add_argument('--lr_', type=str, default='reduce', choices=['step', 'reduce'] )
    parser.add_argument('--lr__', type=float, default=0.00001)
    parser.add_argument('--lr___', type=int, default=25)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--weight', type=str,default='ver7_densenet121_size_512_b18_sche_False_consist_False_valloss_ep120_30_60___best_model_')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--seg_weight', action='store_true', default=False)
    parser.add_argument('--seg_op', type=str,  default='non', choices=['non', 'seg_fast','seg_slow','seg_stop_fast','consist_1','consist_2'])
    parser.add_argument('--feature', type=str,default='_')
    
    parser.add_argument('--st', type=int,default=0)
    parser.add_argument('--de', type=int, default=0)
    parser.add_argument('--clahe', type=float, default=2.0)
    parser.add_argument('--thr', type=float, default=0.24387007)
    
    return parser.parse_args()