import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, strategy="head_only", device = "None")
    """
    load a pretarined ResNet50 model and modify the final layer for our number of classes.
    strategy: 
    - "head_only": freezes all layers except the classifier head final fully connected (dense) layer.
        Useful when you have a small dataset (< 1000 images) and want to leverage pretrained features without overfitting.
    - "last_block": freezes all layers except the final ResNet block (layer4) and the fully connected layer.
        Useful when you have a medium dataset (1000-10,000 images) and want to fine-tune higher-level features.
    - "full": unfreeze everything with layer-wise learning rates.
        Useful when you have a large dataset (> 10,000 images) and want to fine-tune all features/ full adaptation.
    """
