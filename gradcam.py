import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Returns only the classification output (second output)
        _, output = self.model(x)
        return output
    
def create_gradcam(model, layers, DEVICE):
    """
    Wrap the model and select the target layer based on the architecture.
    
    Returns:
        GradCAM: Configured GradCAM instance.
    """
    model_wrapper = ModelWrapper(model)
    model_wrapper.to(DEVICE)
    model_wrapper.eval()

    # Select target layer based on architecture layer argument
    if layers.startswith('densenet'):
        target_layer = [model_wrapper.model.base_model.encoder.features.norm5]
    elif layers.startswith('resnext') or layers.startswith('se_resnext'):
        target_layer = [model_wrapper.model.base_model.encoder.layer4[-1].bn2]
    elif layers.startswith('se_resnet'):
        target_layer = [model_wrapper.model.base_model.encoder.layer4[-1].bn2]
    elif layers == 'inceptionresnetv2':
        target_layer = [model_wrapper.model.base_model.encoder.mixed_7a]
    elif layers.startswith('mit_b'):
        target_layer = [model_wrapper.model.base_model.encoder.blocks[-1].norm1]
    elif layers.startswith('vgg'):
        target_layer = [model_wrapper.model.base_model.encoder.features[-1]]
    elif layers == 'inceptionv4':
        target_layer = [model_wrapper.model.base_model.encoder.features[-1]]
    elif layers.startswith('efficientnet'):
        target_layer = [model_wrapper.model.base_model.encoder._blocks[7][-1]]
    elif layers.startswith('resnet'):
        target_layer = [model_wrapper.model.base_model.encoder.layer4[-1]]
    else:
        target_layer = [model_wrapper.model.base_model.encoder.layer4[-1]]  # default

    return GradCAM(model=model_wrapper, target_layers=target_layer)

def process_gradcam_batch(model, inputs, grad_cam, DEVICE, threshold, rounding_precision=5):
    """
    Process one batch: run inference, compute Grad-CAM, and prepare the visualization.
    
    Returns:
        dict: Contains predicted mask, classification outputs, the input image (numpy),
              and the Grad-CAM overlay.
    """
    inputs = inputs.to(DEVICE)
    
    with torch.no_grad():
        seg_output, cls_pred = model(inputs)
        predicted_mask = (torch.sigmoid(seg_output) >= 0.5).cpu().numpy()
        cls_pred = torch.sigmoid(cls_pred).cpu()
        cls_pred_bin = (cls_pred > threshold).float()
        cls_pred_rounded = round(cls_pred.item(), rounding_precision)
    predicted_mask = np.squeeze(predicted_mask)
    
    # Remove the batch dimension and prepare the input image.
    inputs_squeezed = inputs.squeeze(0)  # shape: (channels, height, width)
    inputs_np = np.transpose(inputs_squeezed.cpu().numpy(), (1, 2, 0))
    
    # Compute Grad-CAM mask.
    grayscale_cam = grad_cam(input_tensor=inputs)
    min_val = np.min(grayscale_cam)
    max_val = np.max(grayscale_cam)
    if max_val - min_val != 0:
        grayscale_cam = np.uint8(255 * (grayscale_cam - min_val) / (max_val - min_val))
    else:
        grayscale_cam = np.zeros_like(grayscale_cam, dtype=np.uint8)
    grayscale_cam = np.squeeze(grayscale_cam)
    
    # Overlay heatmap using OpenCV's COLORMAP_JET.
    colormap = cv2.COLORMAP_JET
    visualization_g = show_cam_on_image(inputs_np, grayscale_cam / 255, use_rgb=True, colormap=colormap)
    
    # Ensure the input image is RGB.
    if inputs_np.ndim == 2 or (inputs_np.ndim == 3 and inputs_np.shape[2] == 1):
        inputs_np = cv2.cvtColor(inputs_np, cv2.COLOR_GRAY2RGB)
    
    return {
        'predicted_mask': predicted_mask,
        'cls_pred_bin': cls_pred_bin,
        'cls_pred_rounded': cls_pred_rounded,
        'inputs_np': inputs_np,
        'visualization_g': visualization_g
    }

# -----------------------------------------
# GradCAM Train
# -----------------------------------------
def generate_gradcam_visualizations_train(model, layers, test_loader, DEVICE, thr_val, save_dir):
    """
    Generate and save Grad-CAM visualizations for training.
    
    This visualization includes multiple subplots:
        1. Original image with label and prediction.
        2. Grad-CAM overlay.
        3. Overlay of image with ground truth mask.
        4. Ground truth mask.
        5. Predicted mask.
    """
    grad_cam = create_gradcam(model, layers, DEVICE)

    # Loop over the test_loader to compute predictions and Grad-CAM images.
    for i, data in enumerate(test_loader):
        inputs = data['image']
        labels = data['label']
        masks = np.squeeze(data['mask'])
        dcm_name = data['dcm_name']

        outputs = process_gradcam_batch(model, inputs, grad_cam, DEVICE, thr_val, rounding_precision=5)

        plt.figure(figsize=(25, 7))

        # Original image with label and prediction.
        plt.subplot(151)
        plt.imshow(outputs['inputs_np'], cmap='gray')
        plt.axis('off')
        plt.title(f"label_{int(labels.item())}_pred_{int(outputs['cls_pred_bin'].item())}")

        # Grad-CAM overlay.
        plt.subplot(152)
        plt.imshow(outputs['visualization_g'])
        plt.axis('off')
        plt.title(f"Grad-CAM_likelihood_{outputs['cls_pred_rounded']}")

        # Overlay: image with ground truth mask.
        plt.subplot(153)
        plt.title(f'{dcm_name},  idx:{i}')
        plt.imshow(outputs['inputs_np'], cmap='gray')
        plt.imshow(masks, cmap='gray', alpha=0.6)
        plt.axis('off')

        # Ground truth mask only.
        plt.subplot(154)
        plt.imshow(masks, cmap='gray')
        plt.axis('off')
        plt.title("Ground_Truth Mask")

        # Predicted mask only.
        plt.subplot(155)
        plt.imshow(outputs['predicted_mask'], cmap='gray')
        plt.axis('off')
        plt.title("Predicted Mask")

        file_name = f"z_label{int(labels.item())}_pred_{int(outputs['cls_pred_bin'].item())}_{i}_{dcm_name[0].split('.')[0]}.png"
        plt.savefig(os.path.join(save_dir, file_name), dpi=400)
        plt.close()

    print("TRAIN: Grad-CAM visualizations saved successfully!")

# -----------------------------------------
# GradCAM Test
# -----------------------------------------
def generate_gradcam_visualizations_test(model, layers, test_loader, DEVICE, threshold, save_dir):
    """
    Generate and save Grad-CAM visualizations for testing.
    
    This visualization creates a single image per case with the predicted class
    and likelihood overlaid.
    """
    grad_cam = create_gradcam(model, layers, DEVICE)

    # Loop through the test loader.
    for i, data in enumerate(test_loader):
        inputs = data['image']
        labels = data['label']
        dcm_name = data['dcm_name']

        outputs = process_gradcam_batch(model, inputs, grad_cam, DEVICE, threshold, rounding_precision=4)

        # Determine predicted class label.
        clspred = 'Pneumoperitoneum' if int(outputs['cls_pred_bin'].item()) == 1 else 'Non Pneumoperitoneum'

        plt.figure(figsize=(7, 7), dpi=114.1)
        plt.imshow(outputs['visualization_g'])
        plt.axis('off')
        plt.title(f"{clspred}\n\nlikelihood:{outputs['cls_pred_rounded']}   thr:{threshold}", fontsize=18)

        file_name = f"{dcm_name[0].split('.')[0]}_ai.png"
        plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight', pad_inches=0.15)
        plt.close()

    print("TEST: Grad-CAM visualizations saved successfully!")