# are these the same for train & test?
# args from argparser in class params; can this be fixed in regards to redundancy?
import segmentation_models_pytorch as smp
import random
import torch
import torch.nn as nn
import os
import numpy as np 

def my_seed_everywhere(seed: int = 42):
    random.seed(seed)  # random
    np.random.seed(seed)  # numpy
    os.environ["PYTHONHASHSEED"] = str(seed)  # os
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multiGPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MultiTaskModel(nn.Module):
    def __init__(self, layers, aux_params, use_cbam=False, reduction_ratio=16, kernel_size=7):
        super().__init__()
        
        # self.use_cbam = use_cbam
        
        self.is_mit_encoder = 'mit' in layers
        
        in_channels = 3 if self.is_mit_encoder else 1
        #print('in_channels :',in_channels)
        self.base_model = smp.Unet(layers, encoder_weights='imagenet', in_channels=in_channels, classes=1, aux_params=aux_params)

        # if self.use_cbam:
        #     self.cbam_block = CBAMBlock(in_channels=self.base_model.encoder.out_channels[-1], 
        #                                 reduction_ratio=reduction_ratio, 
        #                                 kernel_size=kernel_size)

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
    
aux_params=dict(
    pooling='avg',
    dropout=0.5,
    activation=None,
    classes=1,)

model = MultiTaskModel(layers=layers, aux_params=aux_params, use_cbam=cbam_)