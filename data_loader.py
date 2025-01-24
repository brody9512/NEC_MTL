 # CBAMBlock removed
 # args on parameter now
 # CustomDataset_Train is cleaned up, follows args 
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
from monai.data import Dataset
import skimage.io
import skimage.util

# For DICOM
pydicom.config.image_handlers = []

# Lambda
class MyLambda(A_Lambda):
    def __call__(self, force_apply=False, **data):
        return super().__call__(**data)


########## <-- Train --> ##########
class CustomDataset_Train(Dataset):
    def __init__(self, df, args, training=True,apply_voi=False,hu_threshold=None,clipLimit=None,min_side=None):
        
        self.df = df
        self.training = training
        self.apply_voi=apply_voi
        self.hu_threshold = hu_threshold
        self.clipLimit=args.clahe
        self.min_side=args.size
        
        self.transforms = None

        if self.training:
           transforms_list = [
               A.Resize(self.min_side, self.min_side, p=1),  # Resize image

               A.Rotate(limit=args.rotate_angle, p=args.rotate_p),

               A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0.0, p=0.8, border_mode=cv2.BORDER_CONSTANT),

               A.RandomBrightnessContrast(brightness_limit=args.rbc_b, contrast_limit=args.rbc_c, p=args.rbc_p),

               A.RandomGamma(gamma_limit=(args.gamma_min, args.gamma_max), p=args.gamma_p) if args.gamma_t_f else None,
          
               A.GaussNoise(var_limit=(args.gaus_min, args.gaus_max), p=args.gaus_p) if args.gaus_t_f else None,               
           ]

           additional_transforms = []
           
           if args.ela_t_f:
               additional_transforms.append(A.ElasticTransform(alpha=args.ela_alpha, sigma=args.ela_sigma, alpha_affine=args.ela_alpha_aff, p=args.ela_p))
            
           additional_transforms.append(A.CoarseDropout(max_holes=4, max_height=8, max_width=8, fill_value=0, p=0.5))
            
           if additional_transforms:
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

        label = self.df.iloc[idx, self.df.columns.get_loc('label')]

        img_dcm_path = self.df.iloc[idx, self.df.columns.get_loc('img_dcm')]

        img_dcm__0 = pydicom.dcmread(img_dcm_path)

        img_dcm__0.BitsStored = 16

        img_dcm__=img_dcm__0.pixel_array

        img_dcm__ = self.get_pixels_hu(img_dcm__,img_dcm__0)

        x,y,w,h = self.process_row_crop(img_dcm__)
        
        img_dcm__a = self.process_row_crop_coords(img_dcm__, x,y,w,h)
        
        M_new, angle_new,h_new, w_new,threshold =self.process_row_angle(img_dcm__a)
        
        img_dcm__b= self.process_row_angle_ok(img_dcm__a, M_new, angle_new, h_new, w_new)
        
        xmin, xmax, ymin, ymax =self.process_row_angle_ok_background(img_dcm__b, threshold)
        
        img_dcm__c = self.process_row_angle_ok_background_ok(img_dcm__b, xmin, xmax, ymin, ymax)
        
        img_dcm__c = self.normalize(img_dcm__c, option=True)
        
        img_dcm__d = self.resize_and_padding_with_aspect_clahe(img_dcm__c,)
        
        mask_tiff_path = self.df.iloc[idx, self.df.columns.get_loc('mask_tiff')]   

        if pd.isna(mask_tiff_path):
            new_mask = np.full(img_dcm__c.shape, 3)
        else:
            new_mask = tifffile.imread(mask_tiff_path)
       
            if img_dcm__.shape != new_mask.shape:
                new_mask = cv2.resize(new_mask, (img_dcm__.shape[1], img_dcm__.shape[0]))

            new_mask__a = self.process_row_crop_coords(new_mask, x,y,w,h)
            new_mask__b= self.process_row_angle_ok(new_mask__a, M_new, angle_new, h_new, w_new)
            new_mask__cd = self.process_row_angle_ok_background_ok(new_mask__b, xmin, xmax, ymin, ymax)

            if label == 0:
                new_mask_final = np.zeros_like(img_dcm__d)

            else:
                total_elements = new_mask__cd.size        
                                    
                count_black = np.sum((new_mask__cd >= 0) & (new_mask__cd <= 10))
                count_white = np.sum((new_mask__cd >= 245) & (new_mask__cd <= 255))               
                
                if count_black < count_white:

                    new_mask_ = np.zeros_like(new_mask__cd)

                    new_mask_[(new_mask__cd >= 0) & (new_mask__cd <= 10)] = 255

                    new_mask_[(new_mask__cd >= 245) & (new_mask__cd <= 255)] = 0

                    new_mask_final = new_mask_/255
                    
                else:
                    new_mask__cd[(new_mask__cd >= 245) & (new_mask__cd <= 255)] = 255

                    new_mask__cd[(new_mask__cd >= 0) & (new_mask__cd <= 10)] = 0

                    new_mask_final = new_mask__cd / 255

                new_mask_final = albu.PadIfNeeded(min_height=max(new_mask_final.shape), min_width=max(new_mask_final.shape), always_apply=True, border_mode=0)(image=new_mask_final)['image']

        if pd.isna(mask_tiff_path):

            new_mask_final = np.full(img_dcm__d.shape, 3)
            
        dcm_name = os.path.basename(img_dcm_path)

        label = torch.tensor(label, dtype=torch.float32)

        data = self.transforms(image=img_dcm__d, mask=new_mask_final)
        img_dcm__d, new_mask_final = data['image'], data['mask']

        sample = {'image': img_dcm__d, 'mask': new_mask_final, 'label': label,'dcm_name': dcm_name}

        return sample
    

    def __len__(self):
        return len(self.df)
    
    
    def process_row_crop(self, dicom_e):
        dicom_e = cv2.normalize(dicom_e, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dicom_e=dicom_e.astype(np.uint8)

        _, binary = cv2.threshold(dicom_e, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour = max(contours, key=cv2.contourArea)

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

        _, binary_img_new = cv2.threshold(cropped_img, threshold, 255, cv2.THRESH_BINARY_INV)
        binary_img_new = binary_img_new.astype(np.uint8)
        
        kernel = np.ones((7,7),np.uint8)

        binary_img_new = cv2.morphologyEx(binary_img_new, cv2.MORPH_CLOSE, kernel)

        contours_new_, _ = cv2.findContours(binary_img_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_new = max(contours_new_, key=cv2.contourArea)

        rect_new = cv2.minAreaRect(contour_new)

        box_new = cv2.boxPoints(rect_new)

        box_new = np.intp(box_new)

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

        _, binary_rotated_new = cv2.threshold(rotated_img_new, threshold, 255, cv2.THRESH_BINARY_INV)
        binary_rotated_new = binary_rotated_new.astype(np.uint8)

        kernel = np.ones((7,7),np.uint8)
        binary_rotated_new = cv2.morphologyEx(binary_rotated_new, cv2.MORPH_CLOSE, kernel)

        contours_rotated_new, _ = cv2.findContours(binary_rotated_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(binary_rotated_new)
        cv2.drawContours(mask, contours_rotated_new, -1, (255), thickness=cv2.FILLED)
        y, x = np.where(mask == 255)
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        return xmin, xmax, ymin, ymax
   
    def process_row_angle_ok_background_ok(self, img, xmin, xmax, ymin, ymax):
        return img[ymin:ymax+1, xmin:xmax+1] 

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

        try:
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

        if self.hu_threshold is not None:
            img_dcm[img_dcm < self.hu_threshold] = self.hu_threshold
            
        return np.array(img_dcm, dtype=np.int16)
    

    def resize_and_padding_with_aspect_clahe(self, image, ):
        image = np.clip(image, a_min=np.percentile(image, clip_min), a_max=np.percentile(image, clip_max))
        image -= image.min()
        image /= image.max()
        image = skimage.img_as_ubyte(image)
        image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
        
        if self.clipLimit is not None:
            clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(clahe_l,clahe_l))
            image = clahe.apply(image)
        image = skimage.util.img_as_float32(image)
        image = image * 255.
        return image
    
    
########## <-- Test --> ##########
class CustomDataset_Test(Dataset):
    def __init__(self, df, training=True,apply_voi=False,hu_threshold=None,clipLimit=None,min_side=None):
        self.df = df
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
        label = self.df.iloc[idx, self.df.columns.get_loc('label')]

        img_dcm_path = self.df.iloc[idx, self.df.columns.get_loc('img_dcm')]

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
        return len(self.df)

    
    def normalize(self, image, option=False, **kwargs):
        if image.dtype != np.float32:  # Convert the image to float32 if it's not already
            image = image.astype(np.float32)

        if len(np.unique(image)) != 1:
            image -= image.min()
            image /= image.max()

        if option:
            image = (image - 0.5) / 0.5

        return image#.astype('float32')