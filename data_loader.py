# CustomDataset Class requires arguments from args (argparser) values. Could this be fixed?
# CustomDataset is different for train & test code files. Is putting this on a diff file like here on dataset.py even possible?
#   --> put it separately

# only Normalize() function is duplicate so it can be put on a differnt file, but other functions are different between train and test

# segmentation for internal but only classification for external? --> If dataset logic for training vs. testing is significantly different, then
#   separate so = class InternalDataset(...) in dataset_internal.py & class ExternalDataset(...) in dataset_external.py --> YES! 
import albumentations as A
from albumentations import Lambda as A_Lambda
from monai.data import Dataset
import numpy as np


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