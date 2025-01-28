from albumentations import Lambda as A_Lambda
import os
import cv2
import numpy as np
import pandas as pd
import torch
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from torch.utils.data import Dataset
# from monai.data import Dataset
import skimage.io
import skimage.util

class MyLambda(A_Lambda):
    def __call__(self, force_apply=False, **data):
        return super().__call__(**data)

########## <-- Test --> ##########
class CustomDataset_Test(Dataset):
    def __init__(self, df, args, training=True): # ,apply_voi=False,hu_threshold=None,clipLimit=None,min_side=None $$
        self.df = df
        self.training = training
        self.apply_voi=False
        self.hu_threshold = None
        self.clipLimit= self.args.clahe_cliplimit
        self.min_side= args.size
        
        # Build Albumentations transforms
        self.transforms = self.build_transforms()

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 1) Load label and file path
        label = self.df.iloc[idx]['label']
        img_dcm_path = self.df.iloc[idx]['img_dcm']
        dcm_name = os.path.basename(img_dcm_path)

        # 2) Read the image (using OpenCV, grayscale)
        img_dcm = cv2.imread(img_dcm_path, cv2.IMREAD_GRAYSCALE)

        # 3) Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)

        # 4) Apply Albumentations transforms
        transformed = self.transforms(image=img_dcm)
        img_dcm_transformed = transformed['image']

        # 5) Prepare sample dict
        sample = {
            'image': img_dcm_transformed,
            'label': label,
            'dcm_name': dcm_name
        }
        return sample
            
    def build_transforms(self):
        """
        Build Albumentations transform pipeline
        analogous to CustomDataset_Train, but with
        the test dataset's specified augmentations.
        """
        if self.training:
            return A.Compose([
                A.Resize(self.min_side, self.min_side, p=1),
                A.Rotate(limit=45, p=0.8),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.0,
                    rotate_limit=0.0,
                    p=0.8,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(
                        alpha=30,
                        sigma=30 * 0.05,
                        alpha_affine=30 * 0.03,
                        p=0.5
                    ),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.CoarseDropout(
                        max_holes=4,
                        max_height=8,
                        max_width=8,
                        fill_value=0,
                        p=0.5
                    ),
                ], p=0.5),
                MyLambda(image=self.normalize),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.min_side, self.min_side, p=1),
                MyLambda(image=self.normalize),
                ToTensorV2()
            ])
    
    def normalize(self, image, option=False, **kwargs):
        """
        Scale image data into [0..1], optionally into [-1..1].
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # If it's not a constant image
        if len(np.unique(image)) > 1: # != 1
            image -= image.min()
            image /= image.max()

        if option:
            # scale to [-1..1]
            image = (image - 0.5) / 0.5

        return image