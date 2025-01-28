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
# from torch.utils.data import Dataset
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
class CustomDataset_Train(Dataset): # %% go with monai for now and if it doesn't work we will uncomment torch.utils // or vice versa
    """
    Custom Dataset for Multi-task learning (Segmentation + Classification).
    For training or validation modes, with data augmentations.
    In the future, you could split this into dataset_internal.py (for train)
    and dataset_external.py (for test) if the logic significantly differs.
    """
    def __init__(self, df, args, training=True): # ,apply_voi=False,hu_threshold=None,clipLimit=None,min_side=None $$
        
        self.df = df
        self.args = args
        self.training = training
        self.apply_voi = False
        self.hu_threshold = None
        self.clipLimit = self.args.clahe_cliplimit
        self.min_side = self.args.size
        
        # Build Albumentations transforms
        self.transforms = self.build_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # ============ 1) Load label, image path, mask path ============
        label = self.df.iloc[idx]['label']
        img_dcm_path = self.df.iloc[idx]['img_dcm']
        mask_tiff_path = self.df.iloc[idx]['mask_tiff']
        
        # ============ 2) Load DICOM ===============
        dicom_obj = pydicom.dcmread(img_dcm_path)
        # Force BitsStored = 16 if needed
        dicom_obj.BitsStored = 16
        dcm_img = dicom_obj.pixel_array
        dcm_img = self.get_pixels_hu(dcm_img, dicom_obj)
        
        # ============ 3) Pre-processing & custom steps (crop/rotate) =======
        x, y, w, h = self.process_row_crop(dcm_img)
        dcm_cropped = self.process_row_crop_coords(dcm_img, x, y, w, h) 
        M_new, angle_new, hh, ww, threshold = self.process_row_angle(dcm_cropped)
        dcm_rotated = self.process_row_angle_ok(dcm_cropped, M_new, angle_new, hh, ww)
        xmin, xmax, ymin, ymax = self.process_row_angle_ok_background(dcm_rotated, threshold)
        dcm_rotated_cropped = self.process_row_angle_ok_background_ok(dcm_rotated, xmin, xmax, ymin, ymax)
        dcm_rotated_cropped = self.normalize(dcm_rotated_cropped, option=True)
        final_dcm = self.resize_and_padding_with_aspect_clahe(dcm_rotated_cropped)
                                
        # ============ 4) Process mask (or placeholders if mask not available) ==========
        if pd.isna(mask_tiff_path):
            # If mask path is NaN => no segmentation label => fill with 3
            new_mask_final = np.full(final_dcm.shape, 3, dtype=np.float32)
        else:
            raw_mask = tifffile.imread(mask_tiff_path)
            if raw_mask.shape != dcm_img.shape:
                raw_mask = cv2.resize(raw_mask, (dcm_img.shape[1], dcm_img.shape[0]))

            # follow same transform steps
            cropped_mask = self.process_row_crop_coords(raw_mask, x, y, w, h)
            rotated_mask = self.process_row_angle_ok(cropped_mask, M_new, angle_new, hh, ww)
            rotated_cropped_mask = self.process_row_angle_ok_background_ok(rotated_mask, xmin, xmax, ymin, ymax)
            
            if label == 0:
                # For label = 0, fill with zeros
                new_mask_final = np.zeros_like(final_dcm, dtype=np.float32)
            else:
                # Attempt to unify black/white ranges
                new_mask_final = self.cleanup_mask(rotated_cropped_mask)
                # Then pad
                new_mask_final = A.PadIfNeeded(
                    min_height=max(new_mask_final.shape),
                    min_width=max(new_mask_final.shape),
                    always_apply=True,
                    border_mode=0
                )(image=new_mask_final)['image']

        # ============ 5) Albumentations (final) ==============
        data_transformed = self.transforms(image=final_dcm, mask=new_mask_final)
        dcm_final = data_transformed['image']
        mask_final = data_transformed['mask']

        # ============ 6) Prepare output dict =============
        sample = {
            'image': dcm_final, 
            'mask': mask_final, 
            'label': torch.tensor(label, dtype=torch.float32),
            'dcm_name': os.path.basename(img_dcm_path)
        }
        return sample
    
    # -------------------------------------------------------
    #         Dataset Utility Methods
    # -------------------------------------------------------
    def build_transforms(self):
        """Build Albumentations transform pipeline."""
        transforms_list = []
         # => base resizing
        transforms_list.append(A.Resize(self.min_side, self.min_side, p=1.0))
        
        if self.training:
            # => rotate
            transforms_list.append(A.Rotate(limit=self.args.rotate_angle, p=self.args.rotate_percentage))
            transforms_list.append(
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.0,
                    rotate_limit=0.0,
                    p=0.8,
                    border_mode=cv2.BORDER_CONSTANT
                )
            )
            # => brightness/contrast
            transforms_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=self.args.rbc_brightness,
                    contrast_limit=self.args.rbc_contrast,
                    p=self.args.rbc_percentage
                )
            )
            # => random gamma
            if self.args.gamma_t_f:
                transforms_list.append(
                    A.RandomGamma(
                        gamma_limit=(self.args.gamma_min, self.args.gamma_max),
                        p=self.args.gamma_p
                    )
                )
            # => gaussian noise
            if self.args.gaus_truefalse:
                transforms_list.append(
                    A.GaussNoise(
                        var_limit=(self.args.gaus_min, self.args.gaus_max),
                        p=self.args.gaus_percentage
                    )
                )
                
            # => one-of transform group
            additional_transforms = []
            if self.args.elastic_truefalse:
                additional_transforms.append(
                    A.ElasticTransform(
                        alpha=self.args.ela_alpha,
                        sigma=self.args.ela_sigma,
                        alpha_affine=self.args.ela_alpha_aff,
                        p=self.args.ela_p
                    )
                )
            additional_transforms.append(
                A.CoarseDropout(
                    max_holes=4,
                    max_height=8,
                    max_width=8,
                    fill_value=0,
                    p=0.5
                )
            )
            if len(additional_transforms) > 0:
                transforms_list.append(A.OneOf(additional_transforms, p=0.5))

        # => finalize: normalization + totensor
        transforms_list.append(MyLambda(image=self.normalize))
        transforms_list.append(ToTensorV2())

        return A.Compose([t for t in transforms_list if t is not None])
    
    def get_pixels_hu(self, pixel_array, dicom_obj):
        """Convert raw DICOM pixel array to HU if possible."""
        try:
            pixel_array = apply_modality_lut(pixel_array, dicom_obj)
        except:
            pixel_array = pixel_array.astype(np.int16)
            intercept = dicom_obj.RescaleIntercept
            slope = dicom_obj.RescaleSlope
            if slope != 1:
                pixel_array = slope * pixel_array.astype(np.float64)
                pixel_array = pixel_array.astype(np.int16)
            pixel_array += np.int16(intercept)

        if self.apply_voi:
            pixel_array = apply_voi_lut(pixel_array, dicom_obj)

        if self.hu_threshold is not None:
            pixel_array[pixel_array < self.hu_threshold] = self.hu_threshold

        return np.array(pixel_array, dtype=np.int16)

    def normalize(self, image, option=False, **kwargs):
        """Scale from [0..1], optionally shift/scale to [-1..1]."""
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # If it's not a constant image
        if len(np.unique(image)) > 1:
            image -= image.min()
            image /= image.max()

        if option:
            # scale to [-1..1]
            image = (image - 0.5) / 0.5

        return image
    
    def resize_and_padding_with_aspect_clahe(self, image):
        """Clip percentile, apply CLAHE, and pad to square."""
        args = self.args
        image = np.clip(
            image,
            a_min=np.percentile(image, args.clip_min),
            a_max=np.percentile(image, args.clip_max)
        )
        image -= image.min()
        image /= image.max()
        image = skimage.img_as_ubyte(image)

        # pad to square
        image = A.PadIfNeeded(
            min_height=max(image.shape),
            min_width=max(image.shape),
            always_apply=True,
            border_mode=0
        )(image=image)['image']

        if self.clipLimit is not None:
            clahe_obj = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(args.clahe_limit, args.clahe_limit))
            image = clahe_obj.apply(image)

        image = skimage.util.img_as_float32(image) * 255.0
        return image

    def cleanup_mask(self, mask_arr):
        """
        Adjust raw tiff mask so that foreground is 1.0, background 0.0.
        If reversed, flip it back.
        """
        # count black vs white
        count_black = np.sum((mask_arr >= 0) & (mask_arr <= 10))
        count_white = np.sum((mask_arr >= 245) & (mask_arr <= 255))

        if count_black < count_white:
            # reversed
            new_mask = np.zeros_like(mask_arr, dtype=np.float32)
            new_mask[(mask_arr >= 0) & (mask_arr <= 10)] = 255
            # the 245-255 region => 0
            # but effectively it is reversed
            new_mask_final = new_mask / 255.0
        else:
            # new_mask__cd의 값이 245~255인 위치에 255를 할당
            mask_arr[(mask_arr >= 245) & (mask_arr <= 255)] = 255
            # new_mask__cd의 값이 0~10인 위치에 0을 할당
            mask_arr[(mask_arr >= 0) & (mask_arr <= 10)] = 0
            # Scale mask values from [0,255] to [0,1]
            new_mask_final = mask_arr / 255.0

        return new_mask_final.astype(np.float32)

    # ----------- The row-crop/angle methods ------------
    def process_row_crop(self, dcm_img):
        dcm_img_8u = cv2.normalize(dcm_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # inside normalize parapeter: dtype=cv2.CV_8U $$
        _, binary = cv2.threshold(dcm_img_8u, 50, 255, cv2.THRESH_BINARY)
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
        elif 80 < threshold < 100:
            threshold += 40
        elif 100 < threshold < 110:
            threshold += 20
        elif 110 < threshold < 130:
            threshold += 10
        elif 140 < threshold < 150:
            threshold -= 10
        elif 150 < threshold < 160:
            threshold -= 20
        elif threshold > 160:
            threshold -= 30

        _, binary_img = cv2.threshold(cropped_img, threshold, 255, cv2.THRESH_BINARY_INV)
        binary_img = binary_img.astype(np.uint8)
        kernel = np.ones((7,7), np.uint8)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_new = max(contours, key=cv2.contourArea)

        rect_new = cv2.minAreaRect(contour_new)
        angle_new = rect_new[-1]
        if 45 < angle_new < 95:
            angle_new -= 90
        elif 5 < angle_new < 45:
            angle_new -= angle_new/2
        elif -45 < angle_new < -5:
            angle_new += angle_new/2

        h_new, w_new = cropped_img.shape[:2] # (h_new, w_new)
        center_new = (w_new // 2, h_new // 2)
        M_new = cv2.getRotationMatrix2D(center_new, angle_new, 1.0)
        
        return M_new, angle_new, h_new, w_new, threshold

    def process_row_angle_ok(self, img, M, angle, h, w):
        rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated_img

    def process_row_angle_ok_background(self, rotated_img, threshold):
        _, bin_rot = cv2.threshold(rotated_img, threshold, 255, cv2.THRESH_BINARY_INV)
        bin_rot = bin_rot.astype(np.uint8)
        kernel = np.ones((7,7),np.uint8)
        bin_rot = cv2.morphologyEx(bin_rot, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(bin_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(bin_rot)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        y, x = np.where(mask == 255)
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        return xmin, xmax, ymin, ymax

    def process_row_angle_ok_background_ok(self, img, xmin, xmax, ymin, ymax):
        return img[ymin:ymax+1, xmin:xmax+1]