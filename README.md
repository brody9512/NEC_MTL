# MTL-CNN: Multi-task Learning Framework for Joint Segmentation and Classification in Pneumoperitoneum Detection from Medical Imaging

<p align="center"><img width="100%" src="./image/figure_final.png" /></p>

## 💡 Highlights
TBA
## 📄 Paper
TBA

## 💾 Requirements
- torch>=1.13.1
- torchvision>=0.14.1
- segmentation-models-pytorch>=0.3.2
- albumentations>=1.3.0
- monai>=1.1.0
- pydicom>=2.3.1
- tifffile>=2022.8.12
- opencv-python>=4.6.0
- Pillow>=9.0.1
- scikit-image>=0.19.3
- scikit-learn>=1.1.3
- matplotlib>=3.5.3
- tqdm>=4.64.0
- pandas>=1.4.0
- numpy>=1.22.0
- pytorch_grad_cam >= 0.2.1

## 🧱 Repository Structure
```
├── config.py 
├── dataset.py 
├── train.py 
├── test.py 
├── model.py
├── losses.py
├── preprocessing_dcmtopng.py
├── gradcam.py
├── optim.py 
├── utils.py 
├── image 
│ └── figure_final.png 
├── dataset 
│ └── train_dataset.py 
│ └── test_dataset.py
```

## 🗂️ Datasets
TBA

## 📜 Script Example
### Train
```
--gamma_t_f --gamma_min 80 --gamma_max 120 --gamma_p 0.5   --size 1024  --batch 6  --gpu 5  --rotate_angle 30 --rotate_p 0.8  --layers densenet169 --epoch 180  --clip_min 0.5  --clip_max 98.5  --rbc_b 0.05 --rbc_c 0.2 --ela_t_f   --ela_alpha 15 --ela_sigma 0.75 --ela_alpha_aff 0.45  --gaus_t_f  --gaus_min 0  --gaus_max 10  --feature B0   
```
### Test
```
--gpu 0 --size 1024  --layers densenet169  --external  --weight 0410_densenet169_non_ep180_Lreduce_1024_b6_cla2.0_clip0.5_98.5_rota30.0_rbc_b0.05_c0.2_ela_T_alp15.0_sig0.75_aff0.45_ela_p0.25_gaus_T_0.0_10.0_ho_F_gam_T_80.0_120.0_sizec_F_0.8_resic_F_codp_F_epoch_loss_[]  --thr 0.61940747499  
```

## 🎯 Results: 
### Train
| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Normal         | 0.91      | 0.95   | 0.93     | 22      |
| PneumoperiT    | 0.97      | 0.93   | 0.95     | 30      |
| **Accuracy**   |           |        | 0.94     | 52      |
| **Macro Avg**  | 0.94      | 0.94   | 0.94     | 52      |
| **Weighted Avg** | 0.94    | 0.94   | 0.94     | 52      |
> **ROC Curve**: Area = 0.98

### Test
| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Normal         | 0.82      | 0.93   | 0.87     | 214     |
| PneumoperiT    | 0.89      | 0.73   | 0.80     | 164     |
| **Accuracy**   |           |        | 0.84     | 378     |
| **Macro Avg**  | 0.85      | 0.83   | 0.83     | 378     |
| **Weighted Avg** | 0.85    | 0.84   | 0.84     | 378     |
> **ROC Curve**: Area = 0.89

## 📝 Citation
TBA

## 🙋🏻‍♂️ Acknowledgements
We would like to thank **Junseong Lee** ([@junjslee](https://github.com/junjslee)) for his remarkable contributions to refactoring the code structure and setting up this GitHub repository.

