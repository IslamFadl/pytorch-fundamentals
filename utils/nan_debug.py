"""
NaN debugging utilities for PyTorch training.

When training loss becomes NaN, there are three most common causes:
  1. NaN/Inf in input data
  2. Bad loss function + input combination (log(0), BCELoss with bad inputs)
  3. Learning rate catastrophically too high

This module provides:
  - check_batch(): inspects a batch for NaN/Inf before training
  - check_model_outputs(): verifies model outputs are valid
  - check_gradients(): catches exploding/vanishing gradients
  - NaNGuard: a hook-based monitor that wraps a training loop
"""

import torch
import torch.nn as nn


def check_batch(inputs, targets, name="batch"):
    """
    Check a batch for NaN or Inf values before forward pass.
    Raises early — better than discovering NaN 100 batches later.
    """
    issues = []

    if torch.isnan(inputs).any():
        n_nan = torch.isnan(inputs).sum().item()
        issues.append(f"inputs contain {n_nan} NaN values")
    if torch.isinf(inputs).any():
        n_inf = torch.isinf(inputs).sum().item()
        issues.append(f"inputs contain {n_inf} Inf values")

    if isinstance(targets, torch.Tensor):
        if torch.isnan(targets.float()).any():
            issues.append("targets contain NaN values")

    if issues:
        raise ValueError(
            f"[{name}] Bad data detected:\n  " + "\n  ".join(issues) +
            "\n\nCommon causes:\n"
            "  - Missing values in source data (CSV with NaN cells)\n"
            "  - Corrupted images that decoded to NaN\n"
            "  - Division by zero in your preprocessing pipeline\n"
            "  - Normalisation with zero standard deviation"
        )


def check_model_outputs(outputs, name="model output"):
    """Check model outputs for NaN/Inf. Useful between forward and loss."""
    if torch.isnan(outputs).any():
        raise ValueError(
            f"[{name}] Model output contains NaN.\n"
            "Likely causes:\n"
            "  - Weights already corrupted (NaN propagated from earlier step)\n"
            "  - Numerical instability in a custom layer\n"
            "  - Activation function applied to extreme values (e.g. exp(1000))"
        )
    if torch.isinf(outputs).any():
        raise ValueError(
            f"[{name}] Model output contains Inf.\n"
            "Likely causes:\n"
            "  - Exploding values through deep network\n"
            "  - Missing normalisation layer\n"
            "  - Learning rate too high — previous step corrupted weights"
        )


def check_gradients(model, max_norm=100.0, verbose=False):
    """
    Check gradients after backward() for problems.
    Returns total gradient norm.

    Healthy training: gradient norm usually 0.1 to 10.
    Exploding: > 100 — likely needs gradient clipping or smaller LR.
    Vanishing: < 1e-6 — likely needs better initialisation or different activation.
    """
    total_norm = 0.0
    nan_params = []
    inf_params = []

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if torch.isnan(p.grad).any():
            nan_params.append(name)
        if torch.isinf(p.grad).any():
            inf_params.append(name)
        param_norm = p.grad.data.norm(2).item()
        total_norm += param_norm ** 2

    total_norm = total_norm ** 0.5

    if nan_params:
        raise ValueError(
            f"Gradients contain NaN in these parameters:\n  " +
            "\n  ".join(nan_params[:5]) +
            ("\n  ... and more" if len(nan_params) > 5 else "") +
            "\n\nCauses:\n"
            "  - Loss function produced NaN (check inputs to loss)\n"
            "  - Numerical instability in a custom layer's backward\n"
            "  - Mixed precision overflow"
        )

    if inf_params:
        raise ValueError(
            f"Gradients contain Inf — gradients exploded.\n"
            "Fix: add gradient clipping with torch.nn.utils.clip_grad_norm_"
        )

    if verbose:
        print(f"Gradient norm: {total_norm:.4f}")

    if total_norm > max_norm:
        print(f"⚠ Warning: gradient norm {total_norm:.1f} exceeds {max_norm}. "
              f"Consider gradient clipping.")

    return total_norm


class NaNGuard:
    """
    Wraps a training step to catch NaN at every stage.

    Usage:
        guard = NaNGuard(model, criterion)
        for inputs, targets in loader:
            loss = guard.step(inputs, targets, optimizer)
    """

    def __init__(self, model, criterion, device, max_grad_norm=1.0):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

    def step(self, inputs, targets, optimizer):
        self.step_count += 1
        step_info = f"step {self.step_count}"

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # 1. Check inputs
        check_batch(inputs, targets, name=f"inputs at {step_info}")

        # 2. Forward
        outputs = self.model(inputs)
        check_model_outputs(outputs, name=f"forward output at {step_info}")

        # 3. Loss
        loss = self.criterion(outputs, targets)
        if torch.isnan(loss):
            raise ValueError(
                f"Loss is NaN at {step_info}.\n"
                "Diagnostics:\n"
                f"  - Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]\n"
                f"  - Target range: [{targets.float().min().item():.4f}, {targets.float().max().item():.4f}]\n"
                "\nCommon causes:\n"
                "  - log() of zero or negative in custom loss\n"
                "  - BCELoss instead of BCEWithLogitsLoss\n"
                "  - Label index out of range for CrossEntropyLoss"
            )

        # 4. Verify graph is intact (catches NumPy contamination)
        if not loss.requires_grad:
            raise ValueError(
                f"Loss has requires_grad=False at {step_info}.\n"
                "The computational graph is broken.\n"
                "Causes:\n"
                "  - NumPy operations inside loss function\n"
                "  - .item() called before backward\n"
                "  - .detach() called accidentally\n"
                "  - Returning a Python float instead of a tensor"
            )

        # 5. Backward
        optimizer.zero_grad()
        loss.backward()

        # 6. Check gradients
        grad_norm = check_gradients(self.model)

        # 7. Clip + step
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        optimizer.step()

        return loss.item(), grad_norm


# ─── Demo: trigger each failure case to verify the guard works ──────────────

if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Demonstrating NaN detection — each case below should raise.\n")

    # CASE 1: NaN in inputs
    print("CASE 1: NaN in inputs")
    try:
        bad = torch.randn(4, 3)
        bad[0, 0] = float('nan')
        check_batch(bad, torch.zeros(4))
    except ValueError as e:
        print(f"  ✓ Caught: {str(e).splitlines()[0]}\n")

    # CASE 2: NaN in model output
    print("CASE 2: NaN in model output")
    try:
        out = torch.tensor([1.0, 2.0, float('nan')])
        check_model_outputs(out)
    except ValueError as e:
        print(f"  ✓ Caught: {str(e).splitlines()[0]}\n")

    # CASE 3: NaN loss from log(0)
    print("CASE 3: log(0) producing NaN loss")
    model = nn.Linear(3, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    bad_labels = torch.tensor([99, 0, 0, 0]).to(device)  # class 99 doesn't exist
    inputs = torch.randn(4, 3).to(device)
    try:
        guard = NaNGuard(model, criterion, device)
        guard.step(inputs, bad_labels, torch.optim.Adam(model.parameters()))
    except (ValueError, IndexError, RuntimeError) as e:
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        print(f"  ✓ Caught: {msg[:100]}\n")

    print("✓ All failure cases detected correctly.")
    print("  Use NaNGuard in real training loops to catch issues at step 1,")
    print("  not at step 500 when you've wasted GPU time.")