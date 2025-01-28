


'''이거 다시 구현해보고,  --size 1024  --batch 6  
cd workspace/changhyun/nec_ch/v1_pneumoperiT_code/ && python3 MTL_train_infer_clsloss_g_pre_ori_240206.py 

 --gamma_t_f --gamma_min 80 --gamma_max 120 --gamma_p 0.5  --epoch_loss epoch_loss 
cd workspace/changhyun/nec_ch/v1_pneumoperiT_code/ && python3 MTL_train_infer_clsloss_g_pre_ori_240206.py  --gamma_t_f --gamma_min 80 --gamma_max 120 --gamma_p 0.5   --size 1024  --batch 6  --gpu 5  --rotate_angle 30 --rotate_p 0.8  --layers densenet169 --epoch 180  --clip_min 0.5  --clip_max 98.5  --rbc_b 0.05 --rbc_c 0.2 --ela_t_f   --ela_alpha 15 --ela_sigma 0.75 --ela_alpha_aff 0.45  --gaus_t_f  --gaus_min 0  --gaus_max 10  --feature B0   & 
              precision    recall  f1-score   support
      Normal       0.91      0.95      0.93        22
 PneumoperiT       0.97      0.93      0.95        30

    accuracy                           0.94        52
   macro avg       0.94      0.94      0.94        52
weighted avg       0.94      0.94      0.94        52
ROC curve (area = 0.98)


cd workspace/changhyun/nec_ch/v1_pneumoperiT_code/ && python3 MTL_external_240206.py --gpu 0 --size 1024  --layers densenet169  --external  --weight 0410_densenet169_non_ep180_Lreduce_1024_b6_cla2.0_clip0.5_98.5_rota30.0_rbc_b0.05_c0.2_ela_T_alp15.0_sig0.75_aff0.45_ela_p0.25_gaus_T_0.0_10.0_ho_F_gam_T_80.0_120.0_sizec_F_0.8_resic_F_codp_F_epoch_loss_[]  --thr 0.61940747499     &
              precision    recall  f1-score   support
      Normal       0.82      0.93      0.87       214
 PneumoperiT       0.89      0.73      0.80       164

    accuracy                           0.84       378
   macro avg       0.85      0.83      0.83       378
weighted avg       0.85      0.84      0.84       378
ROC curve (area = 0.89)
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import cv2
import os
import random
import os 
from glob import glob
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm as tqdm
from tqdm import trange
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import albumentations as albu
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
import torch
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import Dataset
import albumentations as A
from sklearn.metrics import accuracy_score
import albumentations.augmentations.functional as AF
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
from tqdm import tqdm
import argparse
import monai
from monai.transforms import *
from monai.data import Dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import argparse
import itertools
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
import matplotlib.colors as mcolors


parser = argparse.ArgumentParser(description='Arugment 설명')#parser객체 생성

#parser에 인자 추가시키기, start, end 인자 추가
parser.add_argument('--path', type=str, default='/workspace/changhyun/nec_ch/csv_xlxs/pneumoperiT_modified_0_and_1_No_lateral_external_240306.csv')
#parser.add_argument('--name', type=str)
parser.add_argument('--gpu', type=str,default='2')
parser.add_argument('--optim', type=str,default='adam')

parser.add_argument('--epoch', type=int, default=125)

parser.add_argument('--ver', type=int, default=6)
# parser.add_argument('--st', type=int,default=0)
# parser.add_argument('--de', type=int, default=0)

parser.add_argument('--clahe', type=float, default=2.0)
parser.add_argument('--batch', type=int, default=18)

parser.add_argument('--size', type=int, default=512)
parser.add_argument('--lr_type', type=str, default='reduce', choices=['step', 'reduce'] )
parser.add_argument('--lr_startstep', type=float, default=0.0001)
parser.add_argument('--lr_patience', type=int, default=25)

parser.add_argument('--seg_weight', action='store_true', default=False)
parser.add_argument('--seg_op', type=str,  default='non', choices=['non', 'seg_fast','seg_slow','seg_stop_fast','consist_1','consist_2'])

parser.add_argument('--feature', type=str,default='_')
parser.add_argument('--infer', action='store_true')
parser.add_argument('--external', action='store_true')
parser.add_argument('--weight', type=str,default='ver7_densenet121_size_512_b18_sche_False_consist_False_valloss_ep120_30_60___best_model_')

# parser.add_argument('--cbam', action='store_true')
parser.add_argument('--model_threshold', type=float, default=0.24387007)

parser.add_argument('--half', action='store_true')

parser.add_argument('--layers', type=str,default='densenet121', choices=['densenet121','efficientnet-b3', 'densenet169','densenet201','densenet161','resnext50_32x4d','se_resnet50','se_resnet101','se_resnext50_32x4d', 'se_resnext101_32x4d','resnext101_32x8d', 'inceptionresnetv2', 'mit_b0','mit_b1','mit_b2','mit_b3','resnet101','resnet152','vgg16','vgg19','inceptionv4','mobilenet_v2','resnet50','resnet101','resnext50_32x4d','resnext101_32x4d','inceptionv4','efficientnet-b5','efficientnet-b6','vgg16','vgg19','resnext101_32x8d'])


parser.add_argument('--seed', type=int, default=42)

# how to run: cd workspace/changhyun/nec_ch/v1_pneumoperiT_code/ && python3 MTL_external_240206.py --gpu 1 --layers densenet169  --external  --weight  0211_densenet169_non_ep200_lr_reduce_size_512_b18_cbam_False_clip_0.5_98.5_rotate_30.0_rbc_b0.05_c0.2_ela_True_alpha15.0_sigma0.75_alaff0.45_gausTrue_0.0_10.0_cordrop_False_half_False_3   --thr 0.6588932275 --feature 0331   &


#parse_args()를 통해 parser객체의 인자들 파싱
args = parser.parse_args()


# path = args.path
# layers = args.layers
# gpu = args.gpu
# optim = args.optim
# EPOCHS = args.epoch

# ver = args.ver
# st = args.st ## delete, no need
# de = args.de ## delete, no need

# clipLimit = args.clahe_cliplimit
# train_batch=args.batch

# min_side_=args.size
# lr_type=args.lr_
# lr_startstep=args.lr__
# lr_patience=args.lr___

# seg_weight_= args.seg_weight
# feature=args.feature

# infer=args.infer
# external=args.external
# weight_=args.weight

# cbam_ = args.cbam ## delete, no need
# model_threshold = args.thr

# half= args.half

# seed =args.seed




change_epoch = [0, 15, 28, 40, 55, 68, 75]
ratio = [[5, 5], [5, 5],[5, 5],[1,9] ,[1, 9], [1,9], [1, 9]]

if external:
    name=f'{weight_}_ex_{feature}'
else:
    name=f'{weight_}_'
    
save_dir = f"/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/result/{name}"


if os.path.exists(save_dir): 
    shutil.rmtree(save_dir) 
os.mkdir(save_dir)


os.environ["CUDA_VISIBLE_DEVICES"] = gpu#args.gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def my_seed_everywhere(seed: int = 42):
    random.seed(seed)  # random
    np.random.seed(seed)  # numpy
    os.environ["PYTHONHASHSEED"] = str(seed)  # os
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multiGPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

my_seed = seed_
my_seed_everywhere(my_seed)

# Function to initialize seeds in DataLoader workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


df = pd.read_csv(path_)

# Modify only the paths that start with '/home/brody9512/workspace/'
df['img_dcm'] = df['img_dcm'].apply(lambda x: x.replace('/home/brody9512', '') if x.startswith('/home/brody9512') else x)



if external:
    test_df=df
else:
    train_df = df[df['Mode_1'] == 'train']
    val_df = df[df['Mode_1'] == 'validation']
    test_df = df[df['Mode_1'] == 'test']

if half:
    # Sample half of each DataFrame (assuming each has more than 1 row)
    train_df = train_df.sample(n=len(train_df_full) // 2, random_state=42)
    val_df = val_df.sample(n=len(val_df_full) // 2, random_state=42)
    test_df = test_df.sample(n=len(test_df_full) // 2, random_state=42)




from albumentations import Lambda as A_Lambda

class MyLambda(A_Lambda):
    def __call__(self, force_apply=False, **data):
        return super().__call__(**data)

class CustomDataset(Dataset):
    def __init__(self, data_frame, training=True,apply_voi=False,hu_threshold=None,clipLimit=None,min_side=None):
        self.data_frame = data_frame
        self.training = training
        self.apply_voi=apply_voi
        self.hu_threshold = hu_threshold
        self.clipLimit=clipLimit
        self.min_side=min_side
        
        # Always initialize self.transforms
        self.transforms = None
        #4
        if self.training:
            self.transforms = A.Compose([
                A.Resize(self.min_side, self.min_side, p=1),  # Resize image
        
                # 필수적인 augmentations
                A.Rotate(limit=45, p=0.8),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0.0, p=0.8, border_mode=cv2.BORDER_CONSTANT),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
                # 선택적인 augmentations (약하게 적용)
                A.OneOf([
                    A.ElasticTransform(alpha=30, sigma=30 * 0.05, alpha_affine=30 * 0.03, p=0.5),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.CoarseDropout(max_holes=4, max_height=8, max_width=8, fill_value=0, p=0.5),
                ], p=0.5),
        
                MyLambda(image=self.normalize),  
                ToTensorV2()  # Convert to tensor
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(self.min_side, self.min_side, p=1),  # Resize image
                MyLambda(image=self.normalize),
                ToTensorV2()
                ])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Load label
        label = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('label')]

        img_dcm_path = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('img_dcm')]

        dcm_name = os.path.basename(img_dcm_path)

        # Read the image using OpenCV
        img_dcm__d = cv2.imread(img_dcm_path, cv2.IMREAD_GRAYSCALE)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)

        # Apply transformations
        data = self.transforms(image=img_dcm__d)
        img_dcm__d = data['image']
        # Reshape the image and mask and convert to tensor
        #img_dcm = img_dcm.unsqueeze(0) #기존에 [1,320,320] 인 것에서 [1,1,320,320]이 된다
        #new_mask = new_mask.unsqueeze(0)


        sample = {'image': img_dcm__d,  'label': label,'dcm_name': dcm_name}

        return sample
    

    def __len__(self):
        return len(self.data_frame)

    
    def normalize(self, image, option=False, **kwargs):
        if image.dtype != np.float32:  # Convert the image to float32 if it's not already
            image = image.astype(np.float32)

        if len(np.unique(image)) != 1:
            image -= image.min()
            image /= image.max()

        if option:
            image = (image - 0.5) / 0.5

        return image#.astype('float32')
        
    
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


#loss
from typing import Optional, List
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def soft_dice_score(output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None,) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score

class DiceLoss(_Loss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {'binary', 'multilabel', 'multiclass'}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != 'binary', "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == 'multiclass':
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == 'binary':
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == 'multiclass':
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == 'multilabel':
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)

class Dice_BCE_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function_1 = DiceLoss(mode='binary', from_logits=True)
        self.loss_function_2 = torch.nn.BCEWithLogitsLoss()
        self.dice_weight     = 1.0   
        self.bce_weight      = 1.0   

    def forward(self, y_pred, y_true):
        dice_loss  = self.loss_function_1(y_pred, y_true)
        bce_loss   = self.loss_function_2(y_pred, y_true)

        return self.dice_weight*dice_loss + self.bce_weight*bce_loss

# class Consistency_Loss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.L2_loss  = torch.nn.MSELoss()
#         self.maxpool  = torch.nn.MaxPool2d(kernel_size=16, stride=16, padding=0)
#         self.avgpool  = torch.nn.AvgPool2d(kernel_size=16, stride=16, padding=0)

#     def forward(self, y_cls, y_seg):
#         y_cls = torch.sigmoid(y_cls)  # (B, C)
#         y_seg = torch.sigmoid(y_seg)  # (B, C, H, W)

#         # We have to adjust the segmentation pred depending on classification pred
#         # ResNet50 uses four 2x2 maxpools and 1 global avgpool to extract classification pred. that is the same as 16x16 maxpool and 16x16 avgpool
#         y_seg = self.avgpool(self.maxpool(y_seg)).flatten(start_dim=1, end_dim=-1)  # (B, C)
#         loss  = self.L2_loss(y_seg, y_cls)

#         return loss
    
class Uptask_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_cls     = torch.nn.BCEWithLogitsLoss()
        self.loss_seg     = Dice_BCE_Loss()
        self.loss_rec     = torch.nn.L1Loss()
        # self.loss_consist = Consistency_Loss()


    def forward(self, cls_pred=None, cls_gt=None):
            total = self.loss_cls(cls_pred, cls_gt)
            return total
def create_optim(name, net, lr):
    if name == 'adam':
        optimizer    = torch.optim.Adam(params=net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)

    elif name == 'adamw':
        optimizer    = torch.optim.AdamW(params=net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)
    
    else :
        raise KeyError("Wrong optim name `{}`".format(name))        

    return optimizer

# import torch.nn.functional as F
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         # Use MLP with one hidden layer to get channel attention scores
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
#         )
    
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return torch.sigmoid(out)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         padding = kernel_size // 2
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
    
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return torch.sigmoid(x)

# class CBAMBlock(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
#         super(CBAMBlock, self).__init__()
#         self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
#         self.spatial_attention = SpatialAttention(kernel_size)
    
#     def forward(self, x):
#         x = self.channel_attention(x) * x
#         x = self.spatial_attention(x) * x
#         return x


aux_params=dict(
    pooling='avg',
    dropout=0.5,
    activation=None,
    classes=1,)

class MultiTaskModel(nn.Module):
    def __init__(self, layers, aux_params, use_cbam=False, reduction_ratio=16, kernel_size=7):
        super().__init__()
        
        self.use_cbam = use_cbam
        
        self.is_mit_encoder = 'mit' in layers
        
        in_channels = 3 if self.is_mit_encoder else 1
        #print('in_channels :',in_channels)
        self.base_model = smp.Unet(layers, encoder_weights='imagenet', in_channels=in_channels, classes=1, aux_params=aux_params)

        if self.use_cbam:
            self.cbam_block = CBAMBlock(in_channels=self.base_model.encoder.out_channels[-1], 
                                        reduction_ratio=reduction_ratio, 
                                        kernel_size=kernel_size)

    def forward(self, x):
        
        if self.is_mit_encoder and x.size(1) == 1:
            #print('channel 3')
            x = x.repeat(1, 3, 1, 1)
        
        # Encoder
        features = self.base_model.encoder(x)

        # Bottleneck
        bottleneck_features = features[-1]
        if self.use_cbam:
            bottleneck_features = self.cbam_block(bottleneck_features)

        # Decoder and Segmentation Head
        decoder_features = self.base_model.decoder(*features[:-1], bottleneck_features)
        segmentation_output = self.base_model.segmentation_head(decoder_features)

        # Classifier (Auxiliary)
        if self.base_model.classification_head is not None:
            classification_output = self.base_model.classification_head(bottleneck_features)
        else:
            classification_output = None

        #return classification_output
        return segmentation_output, classification_output


model = MultiTaskModel(layers=layers, aux_params=aux_params, use_cbam=cbam_)


'''
nn.AdaptiveAvgPool2d((1, 1)) : 이 레이어는 입력 텐서의 높이와 너비를 각각 1로 줄입니다. 
즉, 각 채널에 대해 평균값을 계산합니다. 
예를 들어, 입력이 [64, 2, 320, 320]이면 출력은 [64, 2, 1, 1]이 됩니다.
nn.Flatten() : 이 레이어는 모든 차원을 하나의 차원으로 평탄화(flatten)합니다. 예를 들어, 입력이 [64, 2, 1, 1]이면 출력은 [64, 2]가 됩니다.
nn.Linear(self.base_model.encoder.out_channels[-1], aux_params["classes"]) : 이 레이어는 입력에 가중치를 곱하고 편향을 더하는 선형 변환을 수행합니다. 
입력 차원은 self.base_model.encoder.out_channels[-1]이고, 출력 차원은 aux_params["classes"]입니다.
'''





# Count the number of GPUs available.
num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

# If more than one GPU is specified, use DataParallel.

if num_gpus > 1:
    model = nn.DataParallel(model)

model = model.to(DEVICE)




#optimizer= create_optim('adam', model, args) #args 안에 args.lr 옵션 있어야
optimizer= create_optim(optim, model, lr__) #args 안에 args.lr 옵션 있어야

# Define the learning rate scheduler
if lr_ == 'reduce':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=lr_p)

elif lr_ == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

# # Define the learning rate scheduler
# if lr_ == 'reduce':
#     # AUC가 더 클 때 학습률을 줄이도록 'max' 모드를 사용
#     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.3, patience=10)

# elif lr_ == 'step':
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)



def save_checkpoint(model,optimizer,longpath):
      checkpoint = {
                  #'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
      } # save all important stuff
      filename = '{}.pth'.format(longpath)
      torch.save(checkpoint , filename) 


# 라이브러리 임포트
import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import math







if external or infer: 
    checkpoint = torch.load(f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{weight_}.pth')

else:
    checkpoint = torch.load(f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{name}_best_model_.pth')
    
# # 체크포인트에서 state_dict를 로드합니다.
# state_dict = checkpoint['model_state_dict']

# # 'module.' 접두사를 제거합니다. 이 접두사는 DataParallel을 사용하여 모델을 저장했을 때 생깁니다.
# new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# # 수정된 state_dict를 모델에 로드합니다.
# model_.load_state_dict(new_state_dict, strict=False)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])




from sklearn.metrics import accuracy_score
from monai.metrics import DiceMetric



def calculate_auc_ci(y_true, y_probs, n_bootstraps=1000, alpha=0.95):
    bootstrapped_aucs = []
    rng = np.random.RandomState(seed_)

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


# 95% 신뢰 구간(CI) 계산을 위한 함수 정의
def calculate_ci(metric, n, z=1.96):  # z for 95% CI is 1.96
    se = np.sqrt((metric * (1 - metric)) / n)
    ci_lower = metric - z * se
    ci_upper = metric + z * se
    return ci_lower, ci_upper


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
            
            # print("Type of labels:", type(labels)) #<class 'torch.Tensor'>
            # print("Labels:", labels) #tensor([[1.]])
            
            inputs = inputs.to(device) 
            labels = labels.to(device)

            # Make predictions with the model.
            _, cls_pred = model(inputs)
            # print("cls_pred :", cls_pred)
            # print("labels shape:", labels.shape) #torch.Size([1, 1])

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

            # Calculate Dice score and print
            #dice_score = dice_metric(y_pred=seg_pred_bin.unsqueeze(1), y=masks.unsqueeze(1)).mean().item()
            

            # dcm_name =['0930566712.1.2.dcm']
            results[dcm_name[0]] = {'Accuracy': acc}

            y_true.extend(labels.cpu().numpy())
            y_prob.extend(cls_pred.cpu().numpy())


    print('cls_loss:', loss)
    
    return y_true, y_prob,results

# 테스트 데이터에 대한 추론 수행
y_true, y_prob, results = test_inference(model, Uptask_Loss(), test_loader, DEVICE)
# y_true = [array([1.], dtype=float32), array([1.], dtype=float32), ...]
# y_prob = [array([0.6194075], dtype=float32), array([0.99301296], dtype=float32), ...]

# Flatten the list of arrays to a list of scalars
y_true_flat = [item[0] for item in y_true]
y_prob_flat_0 = [item[0] for item in y_prob]

# 소수점 다섯 자리까지 반올림
y_prob_flat = [round(num, 4) for num in y_prob_flat_0]

# print(f'y_true: \n{y_true_flat}\n')
# print(f'y_prob: \n{y_prob_flat}\n')

with open(os.path.join(save_dir,f'results_y_true_prob.txt'), 'w', encoding='utf-8') as f:
    f.write(f'y_true: \n{y_true_flat}\n')
    f.write(f'y_prob: \n{y_prob_flat}\n')

print(f'y_prob: \n{y_prob_flat}\n')

y_true_np = np.array(y_true_flat)
y_prob_np = np.array(y_prob_flat)
np.savez(f"{save_dir}/results_y_true_prob.npz", y_true=y_true_np, y_prob=y_prob_np)
# data = np.load(f"{save_dir}/y_data.npz")
# loaded_y_true = data['y_true']
# loaded_y_prob = data['y_prob']


fpr, tpr, th = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

youden = np.argmax(tpr-fpr)
#print ("Youden index:", th[youden])

# Calculate 95% CI for AUC using bootstrap
ci_lower, ci_upper = calculate_auc_ci(np.array(y_true_flat), np.array(y_prob_flat))


plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f, 95%% CI: %0.2f-%0.2f)" % (roc_auc, ci_lower, ci_upper))
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
#plt.show()

# 그림을 저장
plt.savefig(os.path.join(save_dir,f"roc_curve.png"))
plt.close()

from scipy.stats import t

print(f'Thresold Value: {thr}')


y_pred_1 = []
for prob in y_prob:
    if prob >= thr:#th[youden]: ### Youden Index를 이용한 classification
        y_pred_1.append(1)
    else:
        y_pred_1.append(0)








# 이진화 (threshold = 0.5)
A_pred = (y_prob_np >= thr).astype(int)

# 정확도 (Accuracy) 계산
accuracy_A = accuracy_score(y_true_np, A_pred)

# 민감도 (Sensitivity) 계산 (Recall과 동일)
sensitivity_A = recall_score(y_true_np, A_pred)

# 특이도 (Specificity) 계산
# Specificity = TN / (TN + FP)
conf_matrix_A = confusion_matrix(y_true_np, A_pred)
specificity_A = conf_matrix_A[0, 0] / (conf_matrix_A[0, 0] + conf_matrix_A[0, 1])


# Accuracy CI
ci_accuracy_A = calculate_ci(accuracy_A, len(y_true_np))

# Sensitivity CI
ci_sensitivity_A = calculate_ci(sensitivity_A, np.sum(y_true_np == 1))

# Specificity CI
ci_specificity_A = calculate_ci(specificity_A, np.sum(y_true_np == 0))

# 결과 출력
print(f"Model Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})")
print(f"Model Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})")
print(f"Model Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n")

print("ROC curve (area = %0.4f, 95%% CI: %0.4f-%0.4f)" % (roc_auc, ci_lower, ci_upper))







target_names = ["True Non-PP","True PP"]

report=classification_report(y_true, y_pred_1, target_names=target_names) ### accuracy report

#결과 저장 + dcm acc, fscore 추가
with open(os.path.join(save_dir,f'results.txt'), 'w', encoding='utf-8') as f:
    f.write(f'Classification Report:\n{report}\n')
    f.write(f'Youden index:\n{thr}\n')

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

# 임시 데이터 생성

# Confusion matrix 계산
cm = confusion_matrix(y_true, y_pred_1)

def plot_confusion_matrix(cm, classes,classes_, cmap=plt.cm.Blues,save_path=None):
    plt.figure(figsize=(6, 5))

    # Normalize the confusion matrix by rows (i.e., by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Define a colormap with normalization from 0 to 1
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Plot the confusion matrix with normalization
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap, norm=norm)


    # Colorbar
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes_, rotation=0, fontsize=10)
    plt.yticks(tick_marks, classes, rotation=90, fontsize=10, ha='right')

    fmt = '.2f'
    
    
    thresh = thr
    
    # 행렬의 각 셀에 대해 반복
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # 현재 셀의 값이 해당 행의 전체 합계에 대한 비율을 계산
        percent = cm[i, j] / cm[i, :].sum()
        
        # 퍼센트 표시 (위)
        plt.text(j, i-0.1, f"{percent:.2f}", 
                 horizontalalignment="center",
                 verticalalignment="center",
                 fontsize=20,
                 color="white" if cm_normalized[i, j] > thresh else "black")
                #  color="white" if cm[i, j] > thresh else "black")
                
        
        # 현재 셀에 숫자를 표시 (아래)
        plt.text(j, i+0.1, f"({format(cm[i, j], 'd')})", 
                 horizontalalignment="center",
                 verticalalignment="center",
                 fontsize=20,
                 color="white" if cm_normalized[i, j] > thresh else "black")

    
    #plt.axis('off')
    plt.tight_layout()

    # 이미지 저장
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Confusion matrix 그리기
plot_confusion_matrix(cm, target_names,target_names_1,save_path=os.path.join(save_dir, f"confusion_matrix.png"))#'/path/to/save/image.png







### WE DONT USE GRADCAM and SEE IF IT WORKS
'''
# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image

# class ModelWrapper(nn.Module):
#     def __init__(self, model):
#         super(ModelWrapper, self).__init__()
#         self.model = model

#     def forward(self, x):
#         _, output = self.model(x)
#         return output

# model_ = ModelWrapper(MultiTaskModel().to(DEVICE))

# #model_.train()

# target_layer =[model_.model.base_model.encoder.layer4[-1]]
# grad_cam = GradCAM(model=model_, target_layers=target_layer, use_cuda=True)



# Gradcam
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


#model=MultiTaskModel()
#checkpoint = torch.load(f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{name}_best_model_.pth')


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        #output = self.model(x)
        _ , output = self.model(x)
        return  output

# 이건 주석처리로
# # 체크포인트에서 state_dict를 로드합니다.
# state_dict = checkpoint['model_state_dict']

# # # 'module.' 접두사를 제거합니다. 이 접두사는 DataParallel을 사용하여 모델을 저장했을 때 생깁니다.
# # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# # 수정된 state_dict를 모델에 로드합니다.
# model.load_state_dict(state_dict)
# model=model.to(DEVICE)




model_ = ModelWrapper(model)
# # 체크포인트에서 state_dict를 로드합니다.
# checkpoint = torch.load(f'/workspace/changhyun/nec_ch/v1_pneumoperiT_code/weight/{name}_best_model_.pth')
# #model_.load_state_dict(checkpoint['model_state_dict'])


# # 체크포인트에서 state_dict를 로드합니다.
# state_dict = checkpoint['model_state_dict']

# # 'module.' 접두사를 제거합니다. 이 접두사는 DataParallel을 사용하여 모델을 저장했을 때 생깁니다.
# new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# # 수정된 state_dict를 모델에 로드합니다.
# model_.load_state_dict(new_state_dict, strict=False)

#target_layer =[model_.model.base_model.encoder.layer4[-1]]

# Define target_layer based on the chosen architecture
if layers.startswith('densenet'):
    target_layer = [model_.model.base_model.encoder.features.norm5]
elif layers.startswith('resnext') or layers.startswith('se_resnext'):
    target_layer = [model_.model.base_model.encoder.layer4[-1].bn2]
elif layers.startswith('se_resnet'):
    target_layer = [model_.model.base_model.encoder.layer4[-1].bn2]
elif layers == 'inceptionresnetv2':
    target_layer = [model_.model.base_model.encoder.mixed_7a]
elif layers.startswith('mit_b'):
    target_layer = [model_.model.base_model.encoder.blocks[-1].norm1]
elif layers.startswith('vgg'):
    target_layer = [model_.model.base_model.encoder.features[-1]]
elif layers == 'inceptionv4':
    target_layer = [model_.model.base_model.encoder.features[-1]]
elif layers.startswith('efficientnet'):
    # EfficientNet에서 특정 블록에 접근하려면 인덱스를 사용합니다.
    target_layer = [model_.model.base_model.encoder._blocks[7][-1]]
elif layers.startswith('resnet'):
    target_layer = [model_.model.base_model.encoder.layer4[-1]]
else:
    target_layer = [model_.model.base_model.encoder.layer4[-1]]  # Default layer for unknown architectures

grad_cam = GradCAM(model=model_, target_layers=target_layer)
#score_cam = ScoreCAM(model=model_, target_layers=target_layer, use_cuda=True)

#random_indices = random.sample(range(len(test_loader)), 30)
#random_indices = list(range(5, 7))

for i ,data in enumerate(test_loader):
# random_indices = list(range(18, 20))
# for i in random_indices:
#     data = test_loader.dataset[i]
    inputs, labels, dcm_name = data['image'], data['label'],  data['dcm_name']

    # if int(labels.item()) == 0:
    #     continue 
        
    # print('mask: ', masks)
    # print('labels: ', labels)
    # print('dcm_name: ', dcm_name)

   # 예측된 마스크를 얻는 부분
    inputs = inputs.to(DEVICE)

    
    with torch.no_grad():
        segmentation_output, cls_pred= model(inputs)
        
        predicted_mask = (torch.sigmoid(segmentation_output) >= 0.5).cpu().numpy()
        
        cls_pred = torch.sigmoid(cls_pred).cpu()
        cls_pred_bin = (cls_pred > thr).float()
        cls_pred_rounded = round(cls_pred.item(), 4)
        #acc = accuracy_score(labels.cpu().numpy(), cls_pred_bin.cpu().numpy())
    
    predicted_mask = np.squeeze(predicted_mask)
    
    input_tensor = inputs.to(DEVICE)

    inputs = inputs.squeeze(0) # (channels, height, width)
    #print(f"inputs.shape : {inputs.shape}") # torch.Size([1, 512, 512])
    inputs_np = np.transpose(inputs.cpu().numpy(), (1, 2, 0))
    #print(f"inputs_np.shape : {inputs_np.shape}") # (512, 512, 1)
    grayscale_cam = grad_cam(input_tensor=input_tensor)

    min_val = np.min(grayscale_cam)
    max_val = np.max(grayscale_cam)
    if max_val - min_val != 0:
        grayscale_cam = np.uint8(255 * (grayscale_cam - min_val) / (max_val - min_val))
    else:
        grayscale_cam = np.zeros_like(grayscale_cam, dtype=np.uint8)
    # grayscale_cam = np.uint8(255 * (grayscale_cam - np.min(grayscale_cam)) / np.max(grayscale_cam))
    

    grayscale_cam = np.squeeze(grayscale_cam)

    # Use an OpenCV colormap constant
    colormap = cv2.COLORMAP_JET

    visualization_g = show_cam_on_image(inputs_np, grayscale_cam/255, use_rgb=True, colormap=colormap) #(inputs_np, grayscale_cam/255)
    
    # Convert grayscale to RGB if necessary
    if inputs_np.ndim == 2 or (inputs_np.ndim == 3 and inputs_np.shape[2] == 1):
        inputs_np = cv2.cvtColor(inputs_np, cv2.COLOR_GRAY2RGB)
        
    #plt.show()
    if int(cls_pred_bin.item()) == 1:
        clspred = 'Pneumoperitoneum'
    else:
        clspred = 'Non Pneumoperitoneum'

    plt.figure(figsize=(7, 7), dpi=114.1)  # Set the DPI so that the output is 1024x1024 pixels

    plt.imshow(visualization_g)
    plt.axis('off')
    plt.title(f"{clspred}\n\nlikelihood:{cls_pred_rounded}   thr:0.6194",fontsize=18)

    plt.savefig(os.path.join(save_dir, f"{dcm_name[0].split('.')[0]}_ai.png"), bbox_inches='tight', pad_inches=0.15)
    plt.close()
###
'''