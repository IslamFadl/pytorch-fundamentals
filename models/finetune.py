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

def get_optimizer(model, strategy="head_only", base_lr=1e-3):
    """
    Returns an AdamW optimizer with layer-wise learning rate decay.

    Key concept: earlier layers learned general features (edges, textures)
    during ImageNet pretraining. They need smaller LR — we don't want to
    destroy them. Later layers are more task-specific — slightly higher LR.
    The head is random — needs the highest LR.

    Layer-wise LR decay is a standard practice for fine-tuning
    large models (ViT, DETR, DINOv2) on small datasets.
    """
    if strategy == "head_only":
        # Only head has requires_grad=True — simple case
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr,
            weight_decay=1e-4
        )

    elif strategy in ("last_block", "full"):
        # Layer-wise LR: earlier layers get smaller LR
        # Decay factor: each group gets 10x smaller LR than the next
        param_groups = [
            {
                "params": model.layer1.parameters(),
                "lr": base_lr * 0.01,   # earliest — very small
                "name": "layer1"
            },
            {
                "params": model.layer2.parameters(),
                "lr": base_lr * 0.05,
                "name": "layer2"
            },
            {
                "params": model.layer3.parameters(),
                "lr": base_lr * 0.1,
                "name": "layer3"
            },
            {
                "params": model.layer4.parameters(),
                "lr": base_lr * 0.5,
                "name": "layer4"
            },
            {
                "params": model.fc.parameters(),
                "lr": base_lr,           # head — full LR
                "name": "head"
            },
        ]
        # Filter only trainable params
        for g in param_groups:
            g["params"] = [p for p in g["params"] if p.requires_grad]

        return torch.optim.AdamW(param_groups, weight_decay=1e-4)


def count_trainable_params(model):
    """Utility — shows exactly how many params are being trained."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)\n")
    return trainable, total


if __name__ == "__main__":
    print("=" * 50)
    print("Strategy: head_only")
    model, device = get_model(num_classes=10, strategy="head_only")
    count_trainable_params(model)

    print("=" * 50)
    print("Strategy: last_block")
    model, device = get_model(num_classes=10, strategy="last_block")
    count_trainable_params(model)

    print("=" * 50)
    print("Strategy: full (with layer-wise LR)")
    model, device = get_model(num_classes=10, strategy="full")
    optimizer = get_optimizer(model, strategy="full", base_lr=1e-3)
    count_trainable_params(model)
    print("Optimizer param groups and LRs:")
    for g in optimizer.param_groups:
        if "name" in g:
            print(f"  {g['name']:<10} lr={g['lr']}")