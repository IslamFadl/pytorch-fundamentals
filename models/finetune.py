import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, strategy="head_only", device = "None"):
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
    if device == "None":
        device =torch.device("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available()
                             else "cpu")
    print(f"Using device: {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if strategy == "head_only":
        # Freeze ALL layers
        for param in model.parameters():
            param.requires_grad = False
        # Replace + unfreeze only the classification head
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # model.fc parameters are requires_grad=True by default

    elif strategy == "last_block":
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze layer4 (last ResNet block) + head
        for param in model.layer4.parameters():
            param.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif strategy == "full":
        # Unfreeze everything — use with layer-wise LR in get_optimizer
        for param in model.parameters():
            param.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)
    return model, device