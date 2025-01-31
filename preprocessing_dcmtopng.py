# preprocess .dcm files to .png files for test.py usage
# ex) python preprocess.py 
    # --dicom_folder /path/to/dicoms 
    # --day 240306 
    # --temp_input png1 
    # --output_cpu 24
import numpy as np
import cv2
import os
import os 
from glob import glob
import argparse
import shutil
from PIL import Image
from tqdm import tqdm as tqdm
import albumentations as albu
import pydicom
import pydicom.pixel_data_handlers
import skimage.io
import skimage.util
from sklearn.metrics import accuracy_score
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from tqdm import tqdm
from monai.transforms import *
import pydicom
import multiprocessing
import numpy as np
from glob import glob
import shutil


def process_row_crop(dicom_e):
    dcm_img_8u = cv2.normalize(dicom_e, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(dcm_img_8u, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h


def process_row_crop_coords(img, x, y, w, h):
    return img[y:y+h, x:x+w]   


def process_row_angle(cropped_img):
    
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

    h_new, w_new = cropped_img.shape[:2]
    center_new = (w_new // 2, h_new // 2)
    M_new = cv2.getRotationMatrix2D(center_new, angle_new, 1.0)
    
    return M_new, angle_new, h_new, w_new, threshold

def process_row_angle_ok(cropped_img, M_new, angle_new, h_new, w_new):
    rotated_img_new = cv2.warpAffine(cropped_img, M_new, (w_new, h_new), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_img_new

def process_row_angle_ok_background(rotated_img, threshold):
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

def process_row_angle_ok_background_ok(img, xmin, xmax, ymin, ymax):
    return img[ymin:ymax+1, xmin:xmax+1] 


def normalize(image, option=False, **kwargs):
    """Scale from [0..1], optionally shift/scale to [-1..1]."""
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


def get_pixels_hu(img_dcm,img_dcm0,apply_voi=False,hu_threshold=None ):
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
        
    if apply_voi:
        img_dcm = apply_voi_lut(img_dcm, img_dcm0)
        
    # HU thresholding
    if hu_threshold is not None:
        img_dcm[img_dcm < hu_threshold] = hu_threshold
        
    return np.array(img_dcm, dtype=np.int16)



def resize_and_padding_with_aspect_clahe(image,clipLimit=2.0,clip_min=0.5,clip_max=99.5 ):
    image = np.clip(image, a_min=np.percentile(image, clip_min), a_max=np.percentile(image, clip_max))
    image -= image.min()
    
    # 0으로 나누는 것을 방지하기 위한 조건문 추가
    if image.max() != 0:
        image /= image.max()
    else:
        pass
    
    image = skimage.img_as_ubyte(image)
    image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
    
    if clipLimit is not None:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
        image = clahe.apply(image)
    image = skimage.util.img_as_float32(image)
    image = image * 255.
    return image


def resize_and_padding_with_aspect_clahe_temp_to_png(image, clipLimit=2.0, clip_min=0.5, clip_max=99.5):
    # Clip the image to the desired percentile range
    image = np.clip(image, a_min=np.percentile(image, clip_min), a_max=np.percentile(image, clip_max))
    image -= image.min()
    
    if image.max() != 0:
        image /= image.max()

    # Convert to 8-bit (ubyte) after normalization
    image = skimage.img_as_ubyte(image)
    
    # Convert to grayscale if not already (necessary for CLAHE and some color space conversions)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE
    if clipLimit is not None:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        image = clahe.apply(image)

    # Padding to maintain aspect ratio, if necessary
    image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
    
    return image

def normalize_image(image, clip_min=0.0, clip_max=100.0):
    # Ensure image is in a floating point format for division
    image = image.astype(np.float64) # Convert to float64 for processing
    image -= image.min()
    max_val = image.max()
    
    if max_val != 0:  # Avoid division by zero
        image /= max_val
        
    # Convert to float32 for skimage and normalize to [0, 255] for image saving
    image = skimage.util.img_as_float32(image)
    return (image * 255).astype(np.uint8)  # Convert to uint8 for image saving

def dicom_to_temp_png(dicom_path, output_folder):
    # DICOM 파일 읽기
    img_dcm__0 = pydicom.dcmread(dicom_path)
    img_dcm__0.BitsStored = 16
    img_dcm__ = img_dcm__0.pixel_array

    if img_dcm__0.PhotometricInterpretation == 'MONOCHROME1':
        img_dcm__ = img_dcm__.max() - img_dcm__

    img_dcm__ = get_pixels_hu(img_dcm__, img_dcm__0)
    x, y, w, h = process_row_crop(img_dcm__)
    img_dcm__a = process_row_crop_coords(img_dcm__, x, y, w, h)
    M_new, angle_new, h_new, w_new, threshold = process_row_angle(img_dcm__a)
    img_dcm__b = process_row_angle_ok(img_dcm__a, M_new, angle_new, h_new, w_new)
    xmin, xmax, ymin, ymax = process_row_angle_ok_background(img_dcm__b, threshold)
    img_dcm__c = process_row_angle_ok_background_ok(img_dcm__b, xmin, xmax, ymin, ymax)
    
    # Apply normalization
    img_dcm__d = normalize_image(img_dcm__c)

    # PNG로 임시 저장
    temp_output_path = os.path.join(output_folder, os.path.basename(dicom_path).replace('.dcm', '_temp.png'))
    image_pil = Image.fromarray(img_dcm__d)
    
    image_pil.save(temp_output_path)

    return temp_output_path

def process_and_save_png(png_path, output_folder, clip_min=0.0, clip_max=100.0, clalimit=None):
    img_dcm__c = skimage.io.imread(png_path)    
    img_dcm__d = resize_and_padding_with_aspect_clahe_temp_to_png(image=img_dcm__c, clipLimit=clalimit, clip_min=clip_min, clip_max=clip_max)

    # 최종 PNG로 저장
    final_output_path = os.path.join(output_folder, os.path.basename(png_path).replace('_temp', ''))
    image_pil = Image.fromarray(img_dcm__d.astype(np.uint8))
    image_pil.save(final_output_path)

    # 임시 파일 삭제
    #os.remove(png_path)
    return final_output_path 

def process_png(png_path, args, output_folder):
    # 임시 PNG 파일을 최종 PNG 파일로 처리하고 저장하는 함수를 호출합니다.
    process_and_save_png(png_path, output_folder, args.clip_min, args.clip_max, args.clalimit)

def main():
    """
    1) Parse arguments.
    2) Step1: Convert all DICOMs in 'dicom_folder' to '_temp.png'.
    3) Step2: Convert all '_temp.png' to final PNG with CLAHE and padding.
    """
    parser = argparse.ArgumentParser(description="DICOM to PNG Preprocessing Script")
    parser.add_argument('--dicom_folder', type=str, default='/home/brody9512/workspace/changhyun/nec_ch/nec_external_240117/pneumo_external_pneumo_use/',
                        help='Folder containing .dcm files')
    parser.add_argument('--day', type=str, default='240306',
                        help='Day tag for output folder naming')
    parser.add_argument('--temp_input', type=str, default='png1',
                        help='Temporary stage folder name suffix')
    parser.add_argument('--clip_min', type=float, default=0.0,
                        help='Lower percentile for clipping')
    parser.add_argument('--clip_max', type=float, default=100.0,
                        help='Upper percentile for clipping')
    parser.add_argument('--clalimit', type=float, default=2.0,
                        help='CLAHE limit')
    parser.add_argument('--output_cpu', type=int, default=24,
                        help='Number of CPUs for multiprocessing')
    args = parser.parse_args()
    
    dicom_folder = args.dicom_folder # dicom_folder = '/home/brody9512/workspace/changhyun/nec_ch/nec_external_240117/pneumo_external_pneumo_use/'

    day = args.day # day='240306'
    temp_input = args.temp_input # temp_input = 'png1'
    
    # Step1 output folder
    temp_output_folder = f'/home/brody9512/workspace/changhyun/nec_ch/nec_external_{day}/pneumo_external_pneumo_use_{temp_input}/'
    if os.path.exists(temp_output_folder):
        shutil.rmtree(temp_output_folder)
    os.makedirs(temp_output_folder, exist_ok=True)

    print(f"Step1: Converting .dcm -> _temp.png in: {temp_output_folder}")
    
    # Collect dicom paths
    dicom_paths = glob(os.path.join(dicom_folder, '*.dcm'))
    print(f"Found {len(dicom_paths)} DICOM files.")

    # --------------
    # Step1: DICOM -> temp PNG (multiprocessing)
    # --------------
    pool = multiprocessing.Pool(processes=args.output_cpu)
    # partial function that always uses "temp_output_folder"
    process_dicom_part1 = lambda path: process_dicom_part1(path, temp_output_folder)
    pool.map(process_dicom_part1, dicom_paths)
    pool.close()
    pool.join()
    
    
    # --------------
    # Step2: read _temp.png -> final PNG
    # --------------
    input_stage = 'png2'
    output_folder = f'/home/brody9512/workspace/changhyun/nec_ch/nec_external_{day}/pneumo_external_pneumo_use_{input_stage}_clahe{args.clalimit}/'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    print(f"Step2: Converting _temp.png -> final PNG in: {output_folder}")

    # get list of _temp.png
    temp_png_paths = glob(os.path.join(temp_output_folder, '*_temp.png'))
    print(f"Found {len(temp_png_paths)} temp PNG files.")

    pool2 = multiprocessing.Pool(processes=args.output_cpu)
    def func_mp_part2(png_path):
        process_and_save_png(
            png_path,
            output_folder,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            clipLimit=args.clalimit
        )
    pool2.map(func_mp_part2, temp_png_paths)
    pool2.close()
    pool2.join()

    print("\nAll done. Final PNGs are in:", output_folder)


if __name__ == '__main__':
    main()