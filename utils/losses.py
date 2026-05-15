import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation — handles class imbalance by measuring
    overlap directly rather than per-pixel accuracy.

    Key concept: implemented entirely in PyTorch ops.
    NEVER use NumPy inside a loss function — it detaches the tensor
    from the computational graph, gradients become None, weights
    never update. Always verify: loss.requires_grad should be True.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice  (we minimise, so lower = better overlap)
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Predictions: raw logits (B, C, H, W) for segmentation
        # Apply sigmoid for binary or softmax for multi-class
        predictions = torch.sigmoid(predictions)

        # Flatten spatial dimensions
        predictions = predictions.view(-1)
        targets = targets.view(-1).float()

        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy for class imbalance in classification/segmentation.

    When background = 95% of pixels and foreground = 5%:
    - Standard CE: model learns to predict background everywhere, gets 95% acc
    - Weighted CE: foreground class gets higher weight, model is penalised
      more for missing foreground pixels

    This is exactly what you used at THI for the 0.5% pixel frequency problem.
    """

    def __init__(self, class_weights=None, device=None):
        super().__init__()
        if device is None:
            device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        if class_weights is not None:
            class_weights = torch.tensor(
                class_weights, dtype=torch.float32
            ).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, predictions, targets):
        return self.criterion(predictions, targets)


class CombinedLoss(nn.Module):
    """
    BCE + Dice combined — standard in medical imaging and industrial CV.
    BCE handles per-pixel accuracy, Dice handles class imbalance.
    Alpha controls the balance between the two.
    """

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()  # numerically stable — applies
                                            # sigmoid internally. Never use
                                            # BCELoss + manual sigmoid.
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets.float())
        dice_loss = self.dice(predictions, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


def verify_graph_intact(loss):
    """
    Call this during debugging to verify the computational graph is intact.
    If requires_grad is False, something broke the graph (NumPy, .detach(),
    .item() called inside the loss, returning a Python float, etc.)
    """
    if not loss.requires_grad:
        raise RuntimeError(
            "Loss has requires_grad=False — computational graph is broken.\n"
            "Common causes:\n"
            "  1. NumPy operations inside loss function\n"
            "  2. .item() called before backward()\n"
            "  3. .detach() called accidentally\n"
            "  4. Returning Python float instead of tensor"
        )
    print(f"✓ Graph intact — loss.requires_grad={loss.requires_grad}")


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Testing DiceLoss...")
    pred = torch.randn(4, 1, 256, 256).to(device)
    target = torch.randint(0, 2, (4, 1, 256, 256)).to(device)
    loss_fn = DiceLoss()
    loss = loss_fn(pred, target)
    verify_graph_intact(loss)
    print(f"  Dice loss: {loss.item():.4f}")

    print("\nTesting WeightedCrossEntropyLoss...")
    pred = torch.randn(8, 5).to(device)      # batch=8, classes=5
    target = torch.randint(0, 5, (8,)).to(device)
    # Class 0 is rare — give it 5x weight
    loss_fn = WeightedCrossEntropyLoss(
        class_weights=[5.0, 1.0, 1.0, 1.0, 1.0], device=device
    )
    loss = loss_fn(pred, target)
    verify_graph_intact(loss)
    print(f"  Weighted CE loss: {loss.item():.4f}")

    print("\nTesting CombinedLoss (BCE + Dice)...")
    pred = torch.randn(4, 1, 256, 256).to(device)
    target = torch.randint(0, 2, (4, 1, 256, 256)).to(device)
    loss_fn = CombinedLoss(alpha=0.5)
    loss = loss_fn(pred, target)
    verify_graph_intact(loss)
    print(f"  Combined loss: {loss.item():.4f}")

    print("\n✓ All losses verified — computational graph intact in all cases")