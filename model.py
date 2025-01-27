import segmentation_models_pytorch as smp
import torch.nn as nn

class MultiTaskModel(nn.Module):
    """
    Multi-task model using SMP Unet with optional classification head (aux_params).
    """
    def __init__(self, layers, aux_params):
        super().__init__()
                
        self.is_mit_encoder = 'mit' in layers
        
        in_channels = 3 if self.is_mit_encoder else 1
        self.base_model = smp.Unet(layers, encoder_weights='imagenet', in_channels=in_channels, classes=1, aux_params=aux_params)

    def forward(self, x):
        
        if self.is_mit_encoder and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Encoder
        features = self.base_model.encoder(x)

        # Bottleneck
        bottleneck_features = features[-1]

        # Decoder and Segmentation Head
        decoder_features = self.base_model.decoder(*features[:-1], bottleneck_features)
        segmentation_output = self.base_model.segmentation_head(decoder_features)

        # Classifier (Auxiliary)
        if self.base_model.classification_head is not None:
            classification_output = self.base_model.classification_head(bottleneck_features)
        else:
            classification_output = None

        return segmentation_output, classification_output