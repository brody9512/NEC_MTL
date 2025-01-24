

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import cv2
import os
import xml.etree.cElementTree as ET
import random
import os 
from glob import glob
import shutil
import imutils
from imutils.object_detection import non_max_suppression
from PIL import Image
import math
from pandas import Series, DataFrame
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm as tqdm
import sys
from tqdm import trange
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import albumentations as albu
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms
import torch
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import Dataset
import pydicom
import pydicom.pixel_data_handlers
import skimage.io
import skimage.util
import albumentations as A
from sklearn.metrics import accuracy_score
import tifffile
import albumentations.augmentations.functional as AF
import segmentation_models_pytorch as smp
import functools
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
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
import datetime
import re
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
import matplotlib.colors as mcolors

from pydicom.pixel_data_handlers import gdcm_handler

# Ensure GDCM is used
pydicom.config.image_handlers = [gdcm_handler]


parser = argparse.ArgumentParser(description='Arugment 설명')#parser객체 생성

#parser에 인자 추가시키기, start, end 인자 추가
parser.add_argument('--path', type=str, default='/workspace/changhyun/nec_ch/csv_xlxs/pneumoperiT_modified_n3861_final_20240206_pch.csv')

#parser.add_argument('--name', type=str)
parser.add_argument('--gpu', type=str,default='2')

parser.add_argument('--optim', type=str,default='adam')

parser.add_argument('--epoch', type=int, default=120)

parser.add_argument('--ver', type=int, default=6)

parser.add_argument('--st', type=int,default=0)
parser.add_argument('--de', type=int, default=0)

parser.add_argument('--clahe', type=float, default=2.0)


parser.add_argument('--batch', type=int, default=18)

parser.add_argument('--size', type=int, default=512)
parser.add_argument('--lr_', type=str, default='reduce', choices=['step', 'reduce'] )
parser.add_argument('--lr__', type=float, default=0.00005)
parser.add_argument('--lr___', type=int, default=12)

parser.add_argument('--seg_weight', action='store_true', default=False)
parser.add_argument('--seg_op', type=str,  default='non', choices=['non', 'seg_fast','seg_slow','seg_stop_fast_2','seg_stop_fast_0','seg_stop_fast_1','consist_0','consist_1','consist'])

parser.add_argument('--feature', type=str,default='_')
parser.add_argument('--infer', action='store_true')
parser.add_argument('--external', action='store_true')
parser.add_argument('--weight', type=str,default='ver7_densenet121_size_512_b18_sche_False_consist_False_valloss_ep120_30_60___best_model_')

parser.add_argument('--cbam', action='store_true')
parser.add_argument('--half', action='store_true')


parser.add_argument('--layers', type=str,default='densenet121', choices=['densenet121', 'densenet169','densenet201','densenet161','resnext50_32x4d','se_resnet50','se_resnet101','se_resnext50_32x4d', 'se_resnext101_32x4d','resnext101_32x8d', 'inceptionresnetv2', 'mit_b0','mit_b1','mit_b2','mit_b3','resnet101','resnet152','vgg16','vgg19','inceptionv4','mobilenet_v2','resnet50','resnet101','resnext50_32x4d','resnext101_32x4d','inceptionv4','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','vgg16','vgg19','resnext101_32x8d'])




parser.add_argument('--clip_min', type=float, default=0.5)
parser.add_argument('--clip_max', type=float, default=99.5)

parser.add_argument('--rotate_angle', type=float, default=30) ### 30
parser.add_argument('--rotate_p', type=float, default=0.8)   ###  ??

parser.add_argument('--rbc_b', type=float, default=0.2)
parser.add_argument('--rbc_c', type=float, default=0.2)
parser.add_argument('--rbc_p', type=float, default=0.5)

parser.add_argument('--ela_t_f', action='store_true')
parser.add_argument('--ela_alpha', type=float, default=30)
parser.add_argument('--ela_sigma', type=float, default=1.5)
parser.add_argument('--ela_alpha_aff', type=float, default=0.9)
parser.add_argument('--ela_p', type=float, default=0.25)# 월래 0.25

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

parser.add_argument('--epoch_loss', type=str, default='epoch_loss', choices=['epoch_class_loss', 'epoch_loss'] )
parser.add_argument('--k_size', type=int, default=8, choices=[2,4,8,16,32] )

parser.add_argument('--loss_type', type=str, default='bc_di', choices=['bc_di','bc_iou','bc_tv','fo_di','fo_tv','fo_iou'] )

parser.add_argument('--clahe_limit', type=int, default=8)

parser.add_argument('--seed', type=int, default=42)

#기존예시  
# cd workspace/changhyun/nec_ch/v1_pneumoperiT_code/ && python3 MTL_train_infer_clsloss_pre_ori_240206.py --size 1024  --batch 6  --gpu 4  --rotate_angle 30 --rotate_p 0.8  --layers densenet169 --epoch 200  --clip_min 0.5  --clip_max 98.5  --rbc_b 0.05 --rbc_c 0.2 --ela_t_f   --ela_alpha 15 --ela_sigma 0.75 --ela_alpha_aff 0.45  --gaus_t_f  --gaus_min 0  --gaus_max 10 --epoch_loss epoch_class_loss  --feature A18  --seg_op consist &



#parse_args()를 통해 parser객체의 인자들 파싱
args = parser.parse_args()
path_=args.path

layers=args.layers
print('layers :',layers)


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

# 현재 날짜와 시간을 YYYYMMDD_HHMM 형식으로 가져옵니다.
current_time = datetime.datetime.now().strftime("%m%d")

change_epoch = [0, 100, 120, 135, 160, 170, 175] ## [0, 100, 120, 130, 160, 170, 175]

# #seg천천히 죽이기
# ratio = [[1, 9], [15, 85],[20, 80],[30,70] ,[65, 35], [8,2], [9, 1]]

# #seg빨리 죽이기
# ratio = [[1, 9], [2, 8],[65, 35],[5,5] ,[35, 65], [8,2], [9, 1]]

#cls_weight, seg_weight, consist_weight = weights

if seg_op == 'seg_fast':
    ratio = [[1, 0], [1, 0],[1, 0],[1, 0] ,[1, 0], [1, 0], [1, 0]]
    
elif  seg_op == 'seg_slow':
    ratio = [[5, 5], [5, 5],[5, 5],[3,7] ,[3, 7],[3, 7],[3, 7]]
elif  seg_op == 'consist_0':
    ratio =[[5,5, 0], [5,5, 0],[5,5, 0],[5,0,5] ,[5,0,5],[5,0,5],[5,0,5]]
elif  seg_op == 'consist_1':
    ratio =[[1,9, 5], [2,8, 5],[35,65, 50],[5,5,5] ,[65,35, 50], [8,2,5], [9,1, 5]]
elif  seg_op == 'consist':
    ratio =[[5,5, 5], [5,5, 5],[5,5, 5],[5,5,5] ,[5,5, 5], [5,5,5], [5,5, 5]]
elif  seg_op == 'seg_stop_fast_0':
    ratio = [[5, 5], [5, 5],[5, 5],[7,3] ,[7, 3],[7, 3],[7, 3]]
elif  seg_op == 'seg_stop_fast_1':
    ratio = [[5, 5], [5, 5],[5, 5],[8,2] ,[8, 2],[8, 2],[8, 2]]
elif  seg_op == 'seg_stop_fast_2':
    ratio = [[5, 5], [5, 5],[5, 5],[9,1] ,[9, 1],[9, 1],[9, 1]]

elif  seg_op == 'non':
    ratio = [[5, 5], [5, 5],[5, 5],[5, 5], [5, 5],[5, 5],[5, 5]]



if len(ratio[0]) == 3:
    consist_ = True
else:
    consist_ = False

#print('consist_ :',consist_)  #True

# 폴더 이름 생성
if not (external or infer):

    # 변환 전 변수들을 문자열로 만들기
    ela_t_f_str = "T" if ela_t_f else "F"
    gaus_t_f_str = "T" if gaus_t_f else "F"
    Horizontal_t_f_str = "T" if Horizontal_t_f else "F"
    gamma_t_f_str = "T" if gamma_t_f else "F"
    sizecrop_t_f_str = "T" if sizecrop_t_f else "F"
    resizecrop_t_f_str = "T" if resizecrop_t_f else "F"
    cordrop_t_f_str = "T" if cordrop_t_f else "F"
    half_str = "T" if half else "F"
    
    if epoch_loss_ == 'epoch_loss':
        ep_loss='los'
    elif epoch_loss_ == 'epoch_class_loss':
        ep_loss='clos'

    # 파일 이름 생성
    name = f"{current_time}_{layers}_{seg_op}_{lr_}_{min_side_}_b{train_batch}_cla{clipLimit_}_cli{clip_min}_{clip_max}_rot{rotate_angle}_rbc_b{rbc_b}_c{rbc_c}_ela_{ela_t_f_str}_alp{ela_alpha}_sig{ela_sigma}_aff{ela_alpha_aff}_ela_p{ela_p}_gaus_{gaus_t_f_str}_{gaus_min}_{gaus_max}_ho_{Horizontal_t_f_str}_gam_{gamma_t_f_str}_{gamma_min}_{gamma_max}_sic_{sizecrop_t_f_str}_{sizecrop_min_r}_resic_{resizecrop_t_f_str}_codp_{cordrop_t_f_str}_cl_{clahe_l}_{ep_loss}_{loss_type_}_{feature}"
 
    # # 문자열 내의 'False'와 'True'를 각각 'F'와 'T'로 변환합니다.
    # # 단어 경계(\b)를 사용하여 정확한 단어만 매칭됩니다.
    # name = re.sub(r'\bFalse\b', 'F', name)
    # name = re.sub(r'\bTrue\b', 'T', name)

else:
    name = f'{weight_}' 
save_dir = f"/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/result/{name}"
    

# 추가적인 플래그가 참(True)인 경우에만 이름에 추가
# if is_best:
#     save_dir += "_best"
if infer:
    save_dir += "_infer"
if external:
    save_dir += "_ex"



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

df['mask_tiff'] = df['mask_tiff'].apply(lambda x: x.replace('/home/brody9512', '') if x.startswith('/home/brody9512') else x)


if external:
    test_df=df
else:
    # Filter out rows where 'label' is not 0 or 1
    df_filtered = df[df['label'].isin([0, 1])]
    
    # Splitting the filtered DataFrame into train, validation, and test DataFrames
    train_df = df_filtered[df_filtered['Mode_1'] == 'train']
    val_df = df_filtered[df_filtered['Mode_1'] == 'validation']
    test_df = df_filtered[df_filtered['Mode_1'] == 'test']

# if half:
#     # Sample half of each DataFrame (assuming each has more than 1 row)
#     train_df = train_df.sample(n=len(train_df_full) // 2, random_state=42)
#     val_df = val_df.sample(n=len(val_df_full) // 2, random_state=42)
#     test_df = test_df.sample(n=len(test_df_full) // 2, random_state=42)
    



from albumentations import Lambda as A_Lambda

class MyLambda(A_Lambda):
    def __call__(self, force_apply=False, **data):
        return super().__call__(**data)

class CustomDataset(Dataset):
    def __init__(self, data_frame, training=True,apply_voi=False,hu_threshold=None,clipLimit=None,min_side=None):
        
        self.data_frame = data_frame
        self.training = training ## args.training
        self.apply_voi=apply_voi
        self.hu_threshold = hu_threshold
        self.clipLimit=clipLimit
        self.min_side=min_side ## args.size
        
        # Always initialize self.transforms
        self.transforms = None
        #4
        if self.training:
           transforms_list = [
               A.Resize(self.min_side, self.min_side, p=1),  # Resize image

               A.Rotate(limit=rotate_angle, p=rotate_p),

               A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0.0, p=0.8, border_mode=cv2.BORDER_CONSTANT),


                #brightness_limit=0.2는 이미지의 밝기를 최대 20% 증가 또는 감소 #150이면 120~180
                #이미지의 대비를 최대 20% 증가 또는 감소 # 1.5이면 1.2~1.8
               A.RandomBrightnessContrast(brightness_limit=rbc_b, contrast_limit=rbc_c, p=rbc_p),



               
               
                ######################1
                # RandomGamma : 감마 조정은 이미지의 전반적인 밝기를 조절하지만, 
                #밝기와 대비 조절과는 다른 방식으로 작동합니다. 
                #X-ray 이미지의 경우, 뼈와 조직의 대비를 변경하여 모델이 다양한 노출 수준에서 효과적으로 학습할 수 있게 합니다.

                #감마 보정은 이미지의 픽셀값 감마 값이 100보다 작으면 이미지는 전반적으로 밝아지고, 
                #감마 값이 100보다 크면 이미지는 어두워집니다. 감마 값이 100인 경우, 이미지는 변경되지 않습니다.
               A.RandomGamma(gamma_limit=(gamma_min, gamma_max), p=gamma_p) if gamma_t_f else None,
          





               
                #var_limit=(10.0, 50.0): 이 매개변수는 잡음의 분산 범위
                # 잡음의 분산이 10.0인 경우, 잡음이 상대적으로 약하고 이미지에 미미한 잡음이 추가
               A.GaussNoise(var_limit=(gaus_min, gaus_max), p=gaus_p) if gaus_t_f else None,               
           ]

            
           additional_transforms = []
            #ElasticTransform = 이미지를 뒤틀거나 왜곡시켜 데이터를 더 다양하게
            #Alpha (뒤틀림 강도): 이 매개변수는 이미지에 적용되는 뒤틀림의 강도를 조절
            #Sigma (가우시안 필터 표준 편차): Sigma는 가우시안 필터의 표준 편차를 나타내며, 이미지의 부드러움을 조절합니다. 
            #이 값이 클수록 이미지의 뒤틀림이 더 부드러워지고 형태가 더 부드러워집니다. 
            #alpha_affine 어파인 변환은 이미지를 이동, 회전 및 크기 조정하는 데 사용됩니다. 값이 클수록 어파인 변환의 크기가 커집니다.
            
           if ela_t_f:
               additional_transforms.append(A.ElasticTransform(alpha=ela_alpha, sigma=ela_sigma, alpha_affine=ela_alpha_aff, p=ela_p))
            
           # if cordrop_t_f: #### 여기서 if 적용!!!!
           additional_transforms.append(A.CoarseDropout(max_holes=4, max_height=8, max_width=8, fill_value=0, p=0.5))
            
           if additional_transforms:  # additional_transforms가 비어있지 않은 경우에만 A.OneOf를 추가합니다.
               transforms_list.append(A.OneOf(additional_transforms, p=0.5))


            
           transforms_list.extend([   
               MyLambda(image=self.normalize),
               ToTensorV2(), ])
         
        else:
            
             transforms_list = [
                    A.Resize(self.min_side, self.min_side, p=1),  # Resize image
                    MyLambda(image=self.normalize),
                    ToTensorV2(),]

        self.transforms = A.Compose(transforms_list)





    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Load label
        label = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('label')]

        img_dcm_path = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('img_dcm')]

        #print('img_dcm_path:',img_dcm_path)

        img_dcm__0 = pydicom.dcmread(img_dcm_path)
        # 'Bits Stored' 값을 변경
        img_dcm__0.BitsStored = 16

        img_dcm__=img_dcm__0.pixel_array
        
        #plt.imshow(img_dcm__, cmap='gray')
        img_dcm__ = self.get_pixels_hu(img_dcm__,img_dcm__0)
        
        
        x,y,w,h = self.process_row_crop(img_dcm__)
        #print('process_row_crop_coords:',x,y,w,h)
        
        img_dcm__a = self.process_row_crop_coords(img_dcm__, x,y,w,h)
        #print('img_dcm__a:',img_dcm__a.shape)
        
        M_new, angle_new,h_new, w_new,threshold =self.process_row_angle(img_dcm__a)
        #print('M_new, angle_new,h_new, w_new,threshold:',M_new, angle_new,h_new, w_new,threshold)
        
        img_dcm__b= self.process_row_angle_ok(img_dcm__a, M_new, angle_new, h_new, w_new)
        #print('img_dcm__b:',img_dcm__b.shape)
        
        xmin, xmax, ymin, ymax =self.process_row_angle_ok_background(img_dcm__b, threshold)
        #print('process_row_angle_ok_background:',xmin, xmax, ymin, ymax)
        
        img_dcm__c = self.process_row_angle_ok_background_ok(img_dcm__b, xmin, xmax, ymin, ymax)
        #print('img_dcm__c:',img_dcm__c.shape)
        
        img_dcm__c = self.normalize(img_dcm__c, option=True)
        #print('img_dcm__c.dtype:',img_dcm__c.dtype)
        
        img_dcm__d = self.resize_and_padding_with_aspect_clahe(img_dcm__c,)
        #print('img_dcm__d:',img_dcm__d.shape)
        
        ################################################################################################
  
        # xmin, xmax, ymin, ymax = self.get_cropping_coords(img_dcm__)
        # # 이줄 밑에서 부터는 img_dcm__부터는 다른 특성을 가져야 img_dcm__이름 바꾸기


        mask_tiff_path = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('mask_tiff')]   

        # mask_tiff_path가 NaN인지 확인 # weak label =0,1 있는 것 때문에 해놓음 (형식상 맞추기 위해)
        if pd.isna(mask_tiff_path):
            # img_dcm__c와 동일한 shape의 배열을 생성하고 모든 값을 3으로 채움
            new_mask = np.full(img_dcm__c.shape, 3)
        else:
            new_mask = tifffile.imread(mask_tiff_path)  # read tiff file
       
            # strong label shape가 다른 것 때문에 해놓음
            if img_dcm__.shape != new_mask.shape:
                new_mask = cv2.resize(new_mask, (img_dcm__.shape[1], img_dcm__.shape[0]))


            new_mask__a = self.process_row_crop_coords(new_mask, x,y,w,h)
            new_mask__b= self.process_row_angle_ok(new_mask__a, M_new, angle_new, h_new, w_new)
            new_mask__cd = self.process_row_angle_ok_background_ok(new_mask__b, xmin, xmax, ymin, ymax)




            # # img_dcm와 mask_tiff를 잘라냄
            # img_dcm__ = self.crop_image_using_coords(img_dcm__, xmin, xmax, ymin, ymax)
            # new_mask = self.crop_image_using_coords(new_mask, xmin, xmax, ymin, ymax)

    
            
            
            # strong label 0 때문에 놓음
            if label == 0:
                new_mask_final = np.zeros_like(img_dcm__d)
                #print('new_mask_0:',new_mask) 

            # strong label 1 때문에 놓음
            else:
                # mask_tiff의 전체 요소 수
                total_elements = new_mask__cd.size        
                                    
                count_black = np.sum((new_mask__cd >= 0) & (new_mask__cd <= 10))
                count_white = np.sum((new_mask__cd >= 245) & (new_mask__cd <= 255))               
                

                if count_black < count_white:
                    #print(f"{idx}_count_black:",count_black, "count_white:",count_white) 
                    #print('count_black < count_white 반대로 라벨링 됨')
                    # mask_tiff와 동일한 모양의 0 배열 생성
                    new_mask_ = np.zeros_like(new_mask__cd)

                    # new_mask__cd의 값이 0~10인 위치에 255을 할당
                    new_mask_[(new_mask__cd >= 0) & (new_mask__cd <= 10)] = 255

                    # new_mask__cd의 값이 245~255인 위치에 0를 할당
                    new_mask_[(new_mask__cd >= 245) & (new_mask__cd <= 255)] = 0

                    new_mask_final = new_mask_/255
                    
                else:
                    # new_mask__cd의 값이 245~255인 위치에 255를 할당
                    new_mask__cd[(new_mask__cd >= 245) & (new_mask__cd <= 255)] = 255

                    # new_mask__cd의 값이 0~10인 위치에 0을 할당
                    new_mask__cd[(new_mask__cd >= 0) & (new_mask__cd <= 10)] = 0

                    # Scale mask values from [0,255] to [0,1]
                    new_mask_final = new_mask__cd / 255

                # resize_and_padding_with_aspect_clahe 여기서 pad를 하기 때문에 여기서 해줘야 함
                new_mask_final = albu.PadIfNeeded(min_height=max(new_mask_final.shape), min_width=max(new_mask_final.shape), always_apply=True, border_mode=0)(image=new_mask_final)['image']
                
                #print('new_mask_final_pad :',new_mask_final.shape)

        ########################################################################################
        #  
        # weak label =0,1 있는 것 때문에 해놓음
        if pd.isna(mask_tiff_path):
            # img_dcm__와 동일한 shape의 배열을 생성하고 모든 값을 3으로 채움
            new_mask_final = np.full(img_dcm__d.shape, 3)
            #print('new_mask_final nan :',new_mask_final.shape)

        #print(f"Mask shape: {new_mask.shape}")  # print mask shape
        dcm_name = os.path.basename(img_dcm_path)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)

        # Apply transformations
        data = self.transforms(image=img_dcm__d, mask=new_mask_final)
        img_dcm__d, new_mask_final = data['image'], data['mask']

        # Reshape the image and mask and convert to tensor
        #img_dcm = img_dcm.unsqueeze(0) #기존에 [1,320,320] 인 것에서 [1,1,320,320]이 된다
        #new_mask = new_mask.unsqueeze(0)


        sample = {'image': img_dcm__d, 'mask': new_mask_final, 'label': label,'dcm_name': dcm_name}

        return sample
    

    def __len__(self):
        return len(self.data_frame)
    
    
    def process_row_crop(self, dicom_e):
        dicom_e = cv2.normalize(dicom_e, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dicom_e=dicom_e.astype(np.uint8)
        # 평균보다 낮은 픽셀 값들은 0으로 설정하여 어두운 부분 제거

        
        _, binary = cv2.threshold(dicom_e, 50, 255, cv2.THRESH_BINARY)
        
        #print("binary: ", binary)

        # 이진 이미지에서 연결된 요소 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 가장 큰 컨투어 찾기
        contour = max(contours, key=cv2.contourArea)

        # 컨투어를 기준으로 이미지 잘라내기
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, w, h
    
    
    def process_row_crop_coords(self, img, x, y, w, h):
        return img[y:y+h, x:x+w]   
    
    
    def process_row_angle(self, cropped_img):
        
        threshold = np.mean(cropped_img) 
        
        if threshold < 80:
            threshold += 40
            
        elif 80 < threshold < 90:
            threshold += 40
            
        elif 90 < threshold < 100:
            threshold += 40
            
        elif 100 < threshold < 110:
            threshold += 20

        elif 110 < threshold < 120:
            threshold += 10
            
        elif 120 < threshold < 130:
            threshold += 10
            
        elif 140 < threshold< 150:
            threshold -= 10
            
        elif 150 < threshold < 160:
            threshold -= 20
            
        elif 160 < threshold < 170 :
            threshold -= 30

        elif 170 < threshold :
            threshold -= 30


        # 이미지 이진화
        _, binary_img_new = cv2.threshold(cropped_img, threshold, 255, cv2.THRESH_BINARY_INV)
        binary_img_new = binary_img_new.astype(np.uint8)
        
        kernel = np.ones((7,7),np.uint8)

        # 모폴로지 연산을 통한 노이즈 제거
        binary_img_new = cv2.morphologyEx(binary_img_new, cv2.MORPH_CLOSE, kernel)


        
        # 이진 이미지에서 연결된 요소 찾기
        contours_new_, _ = cv2.findContours(binary_img_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # 가장 큰 컨투어 찾기
        contour_new = max(contours_new_, key=cv2.contourArea)

        # 회전된 바운딩 박스 계산
        rect_new = cv2.minAreaRect(contour_new)
        #print("rect_new: ", rect_new) #rect_new:  ((994.4682006835938, 809.23583984375), (w=1847, h=1139), 79.82633972167969)

        box_new = cv2.boxPoints(rect_new)
        #print("box_new: ", box_new)

        box_new = np.intp(box_new)

        # 이미지 회전
        angle_new = rect_new[-1]


        if 45< angle_new < 95:
            angle_new -= 90

        elif 5< angle_new < 45:
            angle_new -= angle_new/2
            
        elif -45 < angle_new < -5 :
            angle_new += angle_new/2


        (h_new, w_new) = cropped_img.shape[:2]
        center_new = (w_new // 2, h_new // 2)
        M_new = cv2.getRotationMatrix2D(center_new, angle_new, 1.0)
        
        return M_new, angle_new, h_new, w_new, threshold
    
    def process_row_angle_ok(self,cropped_img, M_new, angle_new, h_new, w_new):
        rotated_img_new = cv2.warpAffine(cropped_img, M_new, (w_new, h_new), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated_img_new
    
    def process_row_angle_ok_background(self,rotated_img_new, threshold):
        
        # 회전된 이미지에서 다시 이진화
        _, binary_rotated_new = cv2.threshold(rotated_img_new, threshold, 255, cv2.THRESH_BINARY_INV)
        binary_rotated_new = binary_rotated_new.astype(np.uint8)

        # 모폴로지 연산을 통한 노이즈 제거
        kernel = np.ones((7,7),np.uint8)
        binary_rotated_new = cv2.morphologyEx(binary_rotated_new, cv2.MORPH_CLOSE, kernel)

        # 이진 이미지에서 연결된 요소 찾기
        contours_rotated_new, _ = cv2.findContours(binary_rotated_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(binary_rotated_new)
        cv2.drawContours(mask, contours_rotated_new, -1, (255), thickness=cv2.FILLED)
        y, x = np.where(mask == 255)
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        return xmin, xmax, ymin, ymax
   
    def process_row_angle_ok_background_ok(self, img, xmin, xmax, ymin, ymax):
        return img[ymin:ymax+1, xmin:xmax+1] 



    # def get_cropping_coords(self, img):
    #     img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #     #img = img.astype(np.uint8)
    #     _, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     mask = np.zeros_like(binary)
    #     cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    #     y, x = np.where(mask == 255)
    #     xmin, xmax = np.min(x), np.max(x)
    #     ymin, ymax = np.min(y), np.max(y)
    #     return xmin, xmax, ymin, ymax


    # def crop_image_using_coords(self, img, xmin, xmax, ymin, ymax):
    #     return img[ymin:ymax+1, xmin:xmax+1]


    
    def normalize(self, image, option=False, **kwargs):
        if image.dtype != np.float32:  # Convert the image to float32 if it's not already
            image = image.astype(np.float32)

        if len(np.unique(image)) != 1:
            image -= image.min()
            image /= image.max()

        if option:
            image = (image - 0.5) / 0.5

        return image#.astype('float32')
    


    
    def get_pixels_hu(self, img_dcm,img_dcm0 ):

        #dcm_image.BitsStored = 16 
        # image  = dcm_image.pixel_array
        try:
            #image = apply_modality_lut(image, dcm_image)
            img_dcm = apply_modality_lut(img_dcm, img_dcm0)
        except:
            img_dcm = img_dcm.astype(np.int16)
            intercept = img_dcm0.RescaleIntercept
            slope = img_dcm0.RescaleSlope
            if slope != 1:
                img_dcm = slope * img_dcm.astype(np.float64)
                img_dcm = img_dcm.astype(np.int16)
            img_dcm += np.int16(intercept)
        if self.apply_voi:
            img_dcm = apply_voi_lut(img_dcm, img_dcm0)
            
        # HU thresholding
        if self.hu_threshold is not None:
            img_dcm[img_dcm < self.hu_threshold] = self.hu_threshold
            
        return np.array(img_dcm, dtype=np.int16)
    

    def resize_and_padding_with_aspect_clahe(self, image, ):
        image = np.clip(image, a_min=np.percentile(image, clip_min), a_max=np.percentile(image, clip_max))
        image -= image.min()
        image /= image.max()
        image = skimage.img_as_ubyte(image)
        #if self.training:
        image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
        
        #image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_CUBIC)
        if self.clipLimit is not None:
            clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(clahe_l,clahe_l))
            image = clahe.apply(image)
        image = skimage.util.img_as_float32(image)
        image = image * 255.
        return image


    
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







# '''consist loss '''
class Consistency_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        k_size=k_size_ ## 만액에 size 1024에 k_size= 16이라면!
        final_size = int((min_side_ / k_size** 2))**2  #root 가하는 건 math.sqrt(min_side_)
        self.L2_loss  = torch.nn.MSELoss()
        self.maxpool  = torch.nn.MaxPool2d(kernel_size=k_size, stride=k_size, padding=0)
        self.avgpool  = torch.nn.AvgPool2d(kernel_size=k_size, stride=k_size, padding=0)
        self.fc = nn.Linear(1,final_size)  # 채널 수를 1에서 16로 변경 
        #min_side =1024 , kernel_size = 8    final_size>> 256 
        #min_side =1024 , kernel_size = 16   final_size>> 16

    def forward(self, y_cls, y_seg):
        
        # print('consis_y_cls.shape:',y_cls.shape)
        # print('consis_y_seg.shape:',y_seg.shape)
        
        y_cls = torch.sigmoid(y_cls)  # (B, C)
        y_seg = torch.sigmoid(y_seg)  # (B, C, H, W)
        
        y_cls = self.fc(y_cls)

        # print('sig_y_seg.shape:',y_seg.shape) #sig_y_seg.shape: torch.Size([6, 1, 1024, 1024])        
        # print('sig_y_cls.shape:',y_cls.shape) # sig_y_cls.shape: torch.Size([6, 16])

        # We have to adjust the segmentation pred depending on classification pred
        # ResNet50 uses four 2x2 maxpools and 1 global avgpool to extract classification pred. that is the same as 16x16 maxpool and 16x16 avgpool
        y_seg = self.avgpool(self.maxpool(y_seg)).flatten(start_dim=1, end_dim=-1)  # (B, C)
        
        # print('L2_loss_seg.shape:',y_seg.shape) # L2_loss_seg.shape: torch.Size([6, 16])
        # print('L2_loss_cls.shape:',y_cls.shape) ## L2_loss_cls.shape: torch.Size([6, 16])
        
        loss  = self.L2_loss(y_seg, y_cls)

        return loss



class Uptask_Loss(torch.nn.Module):
    def __init__(self, cls_weight=1.0, seg_weight=1.0, consist_weight=0, loss_type='bc_di'):
        super().__init__()
        # Initialize loss functions as None
        self.loss_cls = None
        self.loss_seg = None
        self.loss_rec = torch.nn.L1Loss()
        self.loss_consist = Consistency_Loss()
        
        # Weights for each component of the loss
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.rec_weight = 1.0
        self.consist_weight = consist_weight
        
        # Select loss type
        self.loss_type = loss_type
        
        if self.loss_type == 'bc_di':
            self.loss_cls = torch.nn.BCEWithLogitsLoss()
            self.loss_seg = Dice_BCE_Loss()

        elif self.loss_type == 'bc_iou':
            self.loss_cls = torch.nn.BCEWithLogitsLoss()  # Assuming we keep BCE for classification
            self.loss_seg = IoULoss()

        elif self.loss_type == 'bc_tv':
            self.loss_cls = torch.nn.BCEWithLogitsLoss()
            self.loss_seg = TverskyLoss()
            
        elif self.loss_type == 'fo_di':
            self.loss_cls = FocalLoss()
            self.loss_seg = Dice_BCE_Loss()
            
        elif self.loss_type == 'fo_tv':
            self.loss_cls = FocalLoss()
            self.loss_seg = TverskyLoss()

        elif self.loss_type == 'fo_iou':
            self.loss_cls = FocalLoss()
            self.loss_seg = IoULoss()
            


    def forward(self, cls_pred=None, seg_pred=None, rec_pred=None, cls_gt=None, seg_gt=None, rec_gt=None, consist=False):
        # If classification prediction and ground truth are provided, calculate classification loss
        loss_cls = self.loss_cls(cls_pred, cls_gt) if cls_pred is not None and cls_gt is not None else 0
        
        # If segmentation prediction and ground truth are provided, calculate segmentation loss
        loss_seg = self.loss_seg(seg_pred, seg_gt) if seg_pred is not None and seg_gt is not None else 0
        
        # If consistency needs to be calculated
        loss_consist = self.loss_consist(cls_pred, seg_pred) if consist else 0
        
        # Combine the losses
        total = self.cls_weight * loss_cls + self.seg_weight * loss_seg + self.consist_weight * loss_consist
        
        # Record the individual components of the loss
        total_ = {'CLS_Loss': (self.cls_weight * loss_cls).item(), 'SEG_Loss': (self.seg_weight * loss_seg).item()}
        if consist:
            total_['Consist_Loss'] = (self.consist_weight * loss_consist).item()
        
        return total, total_


# class Uptask_Loss(torch.nn.Module):
#     def __init__(self, cls_weight=1.0, seg_weight=1.0, consist_weight=0):
#         super().__init__()
#         self.loss_cls = torch.nn.BCEWithLogitsLoss()
#         self.loss_seg = Dice_BCE_Loss()
#         self.loss_rec = torch.nn.L1Loss()
#         self.loss_consist = Consistency_Loss()
        
#         self.cls_weight = cls_weight
#         self.seg_weight = seg_weight
#         self.rec_weight = 1.0
#         self.consist_weight = consist_weight

#     def forward(self, cls_pred=None, seg_pred=None, rec_pred=None, cls_gt=None, seg_gt=None, rec_gt=None, consist=False):
#         loss_cls = self.loss_cls(cls_pred, cls_gt) if cls_pred is not None and cls_gt is not None else 0
        
#         # If either seg_pred or seg_gt is None, calculate only the total loss without segmentation loss
#         loss_seg = self.loss_seg(seg_pred, seg_gt) if seg_pred is not None and seg_gt is not None else 0
#         loss_consist = self.loss_consist(cls_pred, seg_pred) if consist else 0
        
#         total = self.cls_weight * loss_cls + self.seg_weight * loss_seg + self.consist_weight * loss_consist
        
#         total_ = {'CLS_Loss': (self.cls_weight * loss_cls).item(), 
#                   'SEG_Loss': (self.seg_weight * loss_seg).item()}
                  
#         if consist:
#             total_['Consist_Loss'] = (self.consist_weight * loss_consist).item()
        
#         return total, total_



def create_optim(name, net, lr):
    if name == 'adam':
        optimizer    = torch.optim.Adam(params=net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)

    elif name == 'adamw':
        optimizer    = torch.optim.AdamW(params=net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)
    
    else :
        raise KeyError("Wrong optim name `{}`".format(name))        

    return optimizer



aux_params=dict(
    pooling='avg',
    dropout=0.5,
    activation=None,
    classes=1,
)

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


# 멀티 태스크 학습을 위한 Train 함수 정의
def train(model, criterion, data_loader, optimizer, device):
    model.train()
    running_loss = 0
    running_seg_loss = 0
    running_class_loss = 0    
    # 데이터 로더로부터 데이터를 순회합니다.
    for i, data in enumerate(data_loader):
        # 입력, 마스크, 라벨을 가져옵니다.
        inputs, masks, labels = data['image'], data['mask'], data['label']
        
        # 차원을 추가합니다.
        labels = labels.unsqueeze(1)  # labels의 크기를 (2,)에서 (2, 1)로 바꿉니다.
        masks = masks.unsqueeze(1)  # masks의 크기를 (2, 320, 320)에서 (2, 1, 320, 320)로 바꿉니다.
        
        # 데이터를 지정된 디바이스로 이동합니다.
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        
        with torch.set_grad_enabled(True):

            # inputs.shape: torch.Size([2, 1, 320, 320])
            # labels.shape: torch.Size([2,1])
            # masks.shape: torch.Size([2,1, 320, 320])

            # cls_pred.shape: torch.Size([2, 1])
            # seg_pred.shape: torch.Size([2, 1, 320, 320])

            seg_pred, cls_pred = model(inputs)
            #print("seg_pred.shape:", seg_pred.shape)
            #print("cls_pred.shape:", cls_pred.shape)
            #print("labels.shape:", labels.shape)
            
            # three_mask_indices = (masks == 3).any(dim=(1,2,3))
            three_mask_indices = (masks == 3).all(dim=1).all(dim=1).all(dim=1)

            # 3이 포함되지 않은 유효한 마스크의 인덱스를 가져옵니다.
            valid_indices = ~three_mask_indices

            # three_mask_indices에 True가 하나라도 있다면 필터링을 수행합니다.
            if three_mask_indices.any():
                filtered_seg_pred = seg_pred[valid_indices]
                filtered_seg_gt = masks[valid_indices]
            else:
                filtered_seg_pred = seg_pred
                filtered_seg_gt = masks
            
            # print("filtered_seg_pred.shape:", filtered_seg_pred.shape)
            # print("filtered_seg_gt.shape:", filtered_seg_gt.shape)
            # Loss 계산
            loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=filtered_seg_pred, cls_gt=labels, seg_gt=filtered_seg_gt, consist=consist_)
            loss_value = loss.item()

            # 손실 값이 유한한지 확인합니다.
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                return

            # 역전파 및 옵티마이저를 사용하여 가중치를 업데이트합니다.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 평균 손실을 계산합니다.
            running_loss += loss_value * inputs.size(0) #inputs.size(0) batch_size이다
            running_seg_loss += loss_detail['SEG_Loss']* inputs.size(0)
            running_class_loss += loss_detail['CLS_Loss']* inputs.size(0)
#         if i % 3 == 0:
#             loss, sl, cl, current = loss_value, loss_detail['SEG_Loss'], loss_detail['CLS_Loss'], i * len(inputs)
#             print(f"loss: {loss:.4f}, SEG_Loss:{sl:.4f}, CLS_Loss:{cl:.4f}  [{current:>5d}/{len(data_loader.dataset):>5d}]")
    
    epoch_loss = running_loss / len(data_loader.dataset) #len(data_loader.dataset) train 전체 갯수 
    epoch_seg_loss = running_seg_loss / len(data_loader.dataset)
    epoch_class_loss = running_class_loss / len(data_loader.dataset)
    
    sample_loss = {'epoch_loss': epoch_loss, 'epoch_seg_loss': epoch_seg_loss, 'epoch_class_loss': epoch_class_loss}

    print('Train: \n Loss: {:.4f} Seg_Loss: {:.4f} Class_Loss: {:.4f} \n'.format(epoch_loss, epoch_seg_loss, epoch_class_loss))
    
    return sample_loss

from monai.metrics import  DiceMetric, ConfusionMatrixMetric
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score

def test(model, criterion, data_loader, device):
    model.eval()
    
    all_labels = []
    all_preds = []
    
    running_loss = 0
    running_seg_loss = 0
    running_class_loss = 0
    running_consist_loss = 0
    
    confuse_metric = ConfusionMatrixMetric()
    dice_metric = DiceMetric()

    # 데이터 로더로부터 데이터를 순회합니다.
    for i, data in enumerate(data_loader):
        # 입력, 마스크, 라벨을 가져옵니다.
        inputs, masks, labels, dcm_name = data['image'], data['mask'], data['label'],data['dcm_name'] 
        
        # 차원을 추가합니다.
        labels = labels.unsqueeze(1)  # labels의 크기를 (2,)에서 (2, 1)로 바꿉니다.
        masks = masks.unsqueeze(1)  # masks의 크기를 (2, 320, 320)에서 (2, 1, 320, 320)로 바꿉니다.
        
        # 데이터를 지정된 디바이스로 이동합니다.
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        # 모델을 통해 예측을 수행합니다.
        with torch.no_grad():
        #with torch.set_grad_enabled(False):
            seg_pred, cls_pred = model(inputs)
            
            # three_mask_indices = (masks == 3).any(dim=(1,2,3))
            three_mask_indices = (masks == 3).all(dim=1).all(dim=1).all(dim=1)

            # 3이 포함되지 않은 유효한 마스크의 인덱스를 가져옵니다.
            valid_indices = ~three_mask_indices

            # three_mask_indices에 True가 하나라도 있다면 필터링을 수행합니다.
            if three_mask_indices.any():
                filtered_seg_pred = seg_pred[valid_indices]
                filtered_seg_gt = masks[valid_indices]
            else:
                filtered_seg_pred = seg_pred
                filtered_seg_gt = masks


            # Loss 계산
            loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=filtered_seg_pred, cls_gt=labels, seg_gt=filtered_seg_gt, consist=consist_)
            loss_value = loss.item()

            # 손실 값이 유한한지 확인합니다.
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                return

            # 평균 손실을 계산합니다.
            running_loss += loss_value * inputs.size(0)
            running_seg_loss += loss_detail['SEG_Loss'] * inputs.size(0)
            running_class_loss += loss_detail['CLS_Loss'] * inputs.size(0)
            if consist_:
                running_consist_loss += loss_detail['Consist_Loss'] * inputs.size(0)
                
            # post-processing
            cls_pred = torch.sigmoid(cls_pred)
            seg_pred = torch.sigmoid(seg_pred)
        
            # labels: tensor([[1.]], device='cuda:0')
            # cls_pred: tensor([[0.7224]], device='cuda:0')
        
        all_labels.append(labels.cpu().numpy())
        all_preds.append(cls_pred.round().cpu().numpy())  # pred_cls must be round() !!
          
        # Metrics CLS
        confuse_metric(y_pred=cls_pred.round(), y=labels)   # pred_cls must be round() !!

        # Metrics SEG
        dice_metric(y_pred=seg_pred.round(), y=masks)    # pred_seg must be round() !! 
    
    #print('all_labels:',all_labels) #array([[0.]], dtype=float32)....
    #print('all_preds:',all_preds) #array([[2.4057862e-14]], dtype=float32)....
    
    
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    auc = roc_auc_score(all_labels, all_preds)
    
    
    # Compute metrics
    f1 = f1_score(all_labels, all_preds.round())
    acc = accuracy_score(all_labels, all_preds.round())
    sen = recall_score(all_labels, all_preds.round()) # Sensitivity is the same as recall
    spe = precision_score(all_labels, all_preds.round(), zero_division=1)

    dice = dice_metric.aggregate().item()

    confuse_metric.reset()
    dice_metric.reset()
    
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_seg_loss = running_seg_loss / len(data_loader.dataset)
    epoch_class_loss = running_class_loss / len(data_loader.dataset)
    
    if consist_:
        # Calculate and add epoch_consist_loss only if consist_ is True
        epoch_consist_loss = running_consist_loss / len(data_loader.dataset)

        
        sample_loss = {'epoch_loss': epoch_loss, 'epoch_seg_loss': epoch_seg_loss, 'epoch_class_loss': epoch_class_loss, 'epoch_consist_loss': epoch_consist_loss,}
        print('Val: \n Loss: {:.4f} Seg_Loss: {:.4f} Class_Loss: {:.4f}  Consist_Loss: {:.4f} \n'.format(epoch_loss, epoch_seg_loss, epoch_class_loss, epoch_consist_loss))
        
    
    else:
        sample_loss = {'epoch_loss': epoch_loss, 'epoch_seg_loss': epoch_seg_loss, 'epoch_class_loss': epoch_class_loss}
        print('Val: \n Loss: {:.4f} Seg_Loss: {:.4f} Class_Loss: {:.4f} \n'.format(epoch_loss, epoch_seg_loss, epoch_class_loss))


    sample_metrics = {'auc': auc, 'f1': f1, 'acc': acc,'sen': sen,'spe': spe,'dice': dice}

    
    print(' AUC: {:.4f} F1: {:.4f} Acc: {:.4f} Sen: {:.4f} Spe: {:.4f} SEG_Dice: {:.4f} \n'.format(auc, f1, acc, sen, spe, dice))
    
    return sample_loss, sample_metrics


def get_weights_for_epoch(current_epoch, change_epoch, ratio):
    num_weights = len(ratio[0])
    
    for idx, check_epoch in enumerate(change_epoch):
        if current_epoch < check_epoch:
            return np.array(ratio[idx-1]) / np.sum(ratio[idx-1])
    
    # If current_epoch is greater than all values in change_epoch
    return np.array(ratio[-1]) / np.sum(ratio[-1])


if not (external or infer):
 
    lrs =[]
    prev_weights =None
    best_loss = float('inf')
    best_auc = 0.0  # 초기화: 최고의 AUC 값을 저장하기 위한 변수
    
    # 각 손실 및 측정치에 대한 빈 딕셔너리 초기화
    if consist_:
        losses = {k: [] for k in ['train_epoch_loss', 'train_epoch_seg_loss', 'train_epoch_class_loss', 'test_epoch_loss', 'test_epoch_seg_loss', 'test_epoch_class_loss','test_epoch_consist_loss']}

    else:
        losses = {k: [] for k in ['train_epoch_loss', 'train_epoch_seg_loss', 'train_epoch_class_loss', 'test_epoch_loss', 'test_epoch_seg_loss', 'test_epoch_class_loss']}
        
    metrics = {k: [] for k in ['auc', 'f1', 'acc', 'sen', 'spe', 'dice']}

    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n--------------------------------------------------")
    
        if not seg_op =='non':
            # 현재 epoch에 대한 가중치 가져오기
            weights = get_weights_for_epoch(epoch, change_epoch, ratio)
    
            # 이전 가중치와 현재 가중치가 다르면 가중치를 출력합니다.
            if prev_weights is None or not np.array_equal(prev_weights, weights):
                print(f"Weights for Epoch {epoch + 1}: {weights}")
                prev_weights = weights
    
            # 가중치의 길이에 따라 criterion을 설정합니다.
            if len(weights) == 3:
                cls_weight, seg_weight, consist_weight = weights
                train_criterion = Uptask_Loss(cls_weight=cls_weight, seg_weight=seg_weight, consist_weight=consist_weight,loss_type=loss_type_)
                test_criterion = Uptask_Loss(cls_weight=cls_weight, seg_weight=seg_weight, consist_weight=consist_weight,loss_type=loss_type_)
            
            elif len(weights) == 2:
                cls_weight, seg_weight = weights
                train_criterion = Uptask_Loss(cls_weight=cls_weight, seg_weight=seg_weight,loss_type=loss_type_)
                test_criterion = Uptask_Loss(cls_weight=cls_weight, seg_weight=seg_weight,loss_type=loss_type_)
    
        else:
            train_criterion = Uptask_Loss(cls_weight=1.0, seg_weight=1.0,loss_type=loss_type_)
            test_criterion = Uptask_Loss(cls_weight=1.0, seg_weight=1.0,loss_type=loss_type_)
    
        train_criterion = train_criterion.to(DEVICE)
        test_criterion = test_criterion.to(DEVICE)
        
        train_sample_loss = train(model,train_criterion, train_loader, optimizer, DEVICE)
        test_sample_loss, test_sample_metrics = test(model, test_criterion, val_loader, DEVICE)
    
        # 결과를 각각의 딕셔너리에 추가
        for key in losses.keys():
            if 'train' in key:
                losses[key].append(train_sample_loss[key.split('train_')[1]])
            else:
                # value = test_sample_loss.get(key.split('test_')[1], 0)  # Assuming 0 as default; adjust as necessary
                # losses[key].append(value)

                #past
                losses[key].append(test_sample_loss[key.split('test_')[1]])
    
        for key in metrics.keys():
            metrics[key].append(test_sample_metrics[key])
    
        # # 'epoch_class_loss' 대신 'auc'를 기준으로 최적의 모델을 확인하고 저장합니다.
        # if test_sample_metrics['auc'] > best_auc:  # AUC가 더 클 경우 (즉, 더 좋은 경우)
        #     best_auc = test_sample_metrics['auc']
        #     save_checkpoint(model, optimizer, f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{name}_best_model_')
        #     print('Model saved! \n')
    
        # 최적의 모델 저장
        if test_sample_loss[epoch_loss_] < best_loss:
            best_loss = test_sample_loss[epoch_loss_]
            save_checkpoint(model, optimizer, f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{name}')
            print('Model saved! \n')
    
        # scheduler.step(metrics=test_sample_metrics['auc'])  # 'auc'에 따라 학습률 갱신
        # lrs.append(optimizer.param_groups[0]["lr"])
    
        scheduler.step(metrics=test_sample_loss[epoch_loss_])  # test_class_loss에 따라 학습률 갱신
        lrs.append(optimizer.param_groups[0]["lr"])
    
    print("Done!")
    
    #LR
    plt.plot([i+1 for i in range(len(lrs))],lrs,color='g', label='Learning_Rate')
    #plt.show()
    plt.savefig(os.path.join(save_dir,f"LR.png"))
    
    
    
    plt.figure(figsize=(12, 27))
    
    plt.subplot(311)
    # 훈련 데이터에 대한 손실 그래프
    plt.plot(range(EPOCHS), losses['train_epoch_loss'], color='darkred', label='Train Total Loss')
    
    # 테스트 데이터에 대한 손실 그래프
    plt.plot(range(EPOCHS), losses['test_epoch_loss'], color='darkblue', label='Val Total Loss')
    
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("Total Losses", fontsize=16)
    #plt.ylim([0.0, 1.51])
    plt.legend(loc='upper right')

    
    plt.subplot(312)
    # 훈련 데이터에 대한 손실 그래프
    plt.plot(range(EPOCHS), losses['train_epoch_seg_loss'], color='red', label='Train Segmentation Loss')
    plt.plot(range(EPOCHS), losses['train_epoch_class_loss'], color= 'salmon' , label='Train Classification Loss')
    
    # 테스트 데이터에 대한 손실 그래프
    if consist_:
        plt.plot(range(EPOCHS), losses['test_epoch_consist_loss'], color='green', label='Val Consistency Loss')
        
    plt.plot(range(EPOCHS), losses['test_epoch_seg_loss'], color='blue', label='Val Segmentation Loss')
    plt.plot(range(EPOCHS), losses['test_epoch_class_loss'], color='lightblue', label='Val Classification Loss')
    
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("SEG and CLS Losses", fontsize=16)
    #plt.ylim([0.0, 1.51])
    plt.legend(loc='upper right')
    
    
    plt.subplot(313)
    # 훈련 데이터에 대한 메트릭 그래프
    plt.plot(range(EPOCHS), metrics['f1'], color='hotpink', label='F1 Score_(CLS)')
    plt.plot(range(EPOCHS), metrics['dice'], color='royalblue', label='Dice Score_(SEG)')
    
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title("F1(CLS) and Dice Score(SEG)", fontsize=16)
    #plt.ylim([0.0, 100.1])
    plt.legend()
    
    #plt.show()
    plt.savefig(os.path.join(save_dir,f"train_val_loss.png"))
    plt.close()

    
    checkpoint = torch.load(f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{name}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if external or infer: 
    checkpoint = torch.load(f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{weight_}.pth')

else:
    checkpoint = torch.load(f'/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/weight/{name}.pth')
    
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

            #print(f"labels.shape : {labels.shape}") #torch.Size([1, 1])
            #print(f"masks.shape : {masks.shape}") #torch.Size([1, 1, 512, 512])

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


# 테스트 데이터에 대한 추론 수행
y_true, y_prob, avg_cls_loss, avg_seg_loss, dice_score, results = test_inference(model, Uptask_Loss(cls_weight=1.0,seg_weight=1.0).to(DEVICE), test_loader, DEVICE)
# y_true = [array([1.], dtype=float32), array([1.], dtype=float32), ...]
# y_prob = [array([0.6194075], dtype=float32), array([0.99301296], dtype=float32), ...]

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

##########여기서 수정!!!###############
if thr_t_f:
    thr_val = thr
else:
    thr_val = th[youden]

y_pred_1 = []
for prob in y_prob:
    if prob >= thr_val: ### Youden Index를 이용한 classification
        y_pred_1.append(1)
    else:
        y_pred_1.append(0)










# 이진화 (threshold = 0.5)
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





# target_names = ["Non Pneumoperitoneum","Pneumoperitoneum"]
target_names = ["True Non-PP","True PP"]


report=classification_report(y_true, y_pred_1, target_names=target_names) ### accuracy report

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





import matplotlib.colors as mcolors

target_names = ["True Non-PP","True PP"]
target_names_1=["Pred Non-PP","Pred PP"]
# 임시 데이터 생성

# Confusion matrix 계산
cm = confusion_matrix(y_true, y_pred_1)

def plot_confusion_matrix(cm, classes,classes_, title='Confusion matrix', cmap=plt.cm.Blues,save_path=None):
    plt.figure(figsize=(6, 5))

    # Normalize the confusion matrix by rows (i.e., by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Define a colormap with normalization from 0 to 1
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Plot the confusion matrix with normalization
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap, norm=norm)
    plt.title(title)

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















print(f'Classification Report:\n{report}')
print("ROC curve (area = %0.2f)" % auc(fpr, tpr),'\n')

print( f'weight : \n{name}\n' )
print(f'Thresold Value : {thr_val}')

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
        _, output = self.model(x)
        return output

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
    # target_layer = [model_.model.base_model.encoder.get_block('7')]
    target_layer = [model_.model.base_model.encoder._blocks[7][-1]]
elif layers.startswith('resnet'):
    target_layer = [model_.model.base_model.encoder.layer4[-1]]
else:
    target_layer = [model_.model.base_model.encoder.layer4[-1]]  # Default layer for unknown architectures

grad_cam = GradCAM(model=model_, target_layers=target_layer)
#grad_cam = GradCAM(model=model_, target_layers=target_layer, use_cuda=True)
#score_cam = ScoreCAM(model=model_, target_layers=target_layer, use_cuda=True)

#random_indices = random.sample(range(len(test_loader)), 30)
#random_indices = list(range(5, 7))




for i ,data in enumerate(test_loader):
# random_indices = list(range(18, 20))
# for i in random_indices:
#     data = test_loader.dataset[i]
    inputs, labels, masks, dcm_name = data['image'], data['label'], data['mask'], data['dcm_name']

    # if int(labels.item()) == 0:
    #     continue 
        
    # print('mask: ', masks)
    # print('labels: ', labels)
    # print('dcm_name: ', dcm_name)

   # 예측된 마스크를 얻는 부분
    inputs = inputs.to(DEVICE)
    #inputs = inputs.unsqueeze(0)
    
    with torch.no_grad():
        segmentation_output, cls_pred= model(inputs)
        # predicted_mask = torch.sigmoid(segmentation_output).cpu().numpy()
        # predicted_mask[predicted_mask >= 0.5] = 1
        # predicted_mask[predicted_mask < 0.5] = 0
        predicted_mask = (torch.sigmoid(segmentation_output) >= 0.5).cpu().numpy()

        
        cls_pred = torch.sigmoid(cls_pred)
        cls_pred_bin = (cls_pred > thr_val).float()
        cls_pred_rounded = round(cls_pred.item(), 5)
        #acc = accuracy_score(labels.cpu().numpy(), cls_pred_bin.cpu().numpy())
        
    
    predicted_mask = np.squeeze(predicted_mask)
    masks = np.squeeze(masks)

    input_tensor = inputs.to(DEVICE)

    #print(f"predicted_mask.shape : {predicted_mask.shape}") # (512, 512)
    #print(f"input_tensor.shape : {input_tensor.shape}") # torch.Size([1, 1, 512, 512])
    #print(f"inputs.shape : {inputs.shape}") # torch.Size([1, 1, 512, 512])

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


    

    plt.figure(figsize=(25,7))

    plt.subplot(151)
    plt.imshow(inputs_np, cmap='gray')
    # plt.imshow(masks, cmap='jet', alpha=0.6)
    plt.axis('off')
    plt.title(f"label_{int(labels.item())}_pred_{int(cls_pred_bin.item())}")
    
    plt.subplot(152)
    plt.imshow(visualization_g, )
    plt.axis('off')
    plt.title(f"Grad-CAM_likelihood_{cls_pred_rounded}")


    plt.subplot(153)
    plt.title(f'{dcm_name},  idx:{i}')
    plt.imshow(inputs_np, cmap='gray')
    plt.imshow(masks, cmap='gray', alpha=0.6)
    plt.axis('off')

    
    # 예측된 마스크를 시각화하는 부분
    plt.subplot(154)
    plt.imshow(masks, cmap='gray') #, vmin=0, vmax=1)  # Ensure binary mask is displayed in grayscale
    plt.axis('off')
    plt.title("Ground_Truth Mask")

    # 예측된 마스크를 시각화하는 부분
    plt.subplot(155)
    plt.imshow(predicted_mask, cmap='gray') #, vmin=0, vmax=1)  # Ensure binary mask is displayed in grayscale
    plt.axis('off')
    plt.title("Predicted Mask")

    #plt.show()
    plt.savefig(os.path.join(save_dir,f"z_label{int(labels.item())}_pred_{int(cls_pred_bin.item())}_{i}_{dcm_name[0].split('.')[0]}.png"), dpi=400)
    plt.close()


















##