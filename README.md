# PyTorch Fundamentals for Deep Learning

Clean implementations of the PyTorch building blocks I use daily as a perception engineer. Each module is reusable, documented, and demonstrates the correct way to handle the most common failure modes in deep learning training pipelines.

This repo is part of a larger portfolio of robotics and CV projects — see [my GitHub profile](https://github.com/IslamFadl) for the full set.

---

## Why this exists

Most PyTorch tutorials show the working "happy" path. This repo shows the issues when you modify or change losses, class distibution, hyperparameters, and more. it also shows how to keep the GPU fed, how to debug NaN losses at step 1 instead of step 50, how to fine-tune large pretrained models on small datasets without destroying their representations, and how to write a custom loss that doesn't break the computational graph.

---

## Structure
```
pytorch-fundamentals/
├── data/
│   └── dataset.py              # Reusable ImageClassificationDataset with train/val transforms
├── models/
│   └── finetune.py             # Fine-tuning strategies with layer-wise LR decay
├── utils/
    ├── dataloader_utils.py     # DataLoader benchmark — measures num_workers impact
    ├── trainer.py              # Training loop with correct train/eval/no_grad pattern
    ├── losses.py               # DiceLoss, WeightedCE, CombinedLoss with graph verification
    └── nan_debug.py            # NaNGuard and checks for the three most common NaN causes
```
---

## Key concepts demonstrated

### 1. The correct training and validation loop

A common bug: forgetting to call `model.eval()` before validation. BatchNorm continues using batch statistics instead of running statistics, Dropout stays active, and validation results become noisy and wrong. Equally common: confusing `model.eval()` with `torch.no_grad()` — they do different things:

```python
model.train()                       # BatchNorm uses batch stats, Dropout active
for batch in train_loader:
    ...

model.eval()                        # BatchNorm uses running stats, Dropout off
with torch.no_grad():               # Disables gradient computation entirely
    for batch in val_loader:
        ...
```

See `utils/trainer.py` for the full implementation.

### 2. DataLoader benchmark — keeping the GPU fed

A common failure mode: GPU utilisation drops to 0% periodically and training is slower than expected. The root cause is usually a DataLoader bottleneck — the CPU can't load and decode batches fast enough to keep the GPU busy. The fix is `num_workers > 0`, `pin_memory=True` on CUDA, and `prefetch_factor`.

The benchmark script measures the actual speedup on your hardware:

| num_workers | pin_memory | time (s) | batches/s |
|-------------|------------|----------|-----------|
| 0           | False      | 0.116    | 431       |
| 2           | False      | 0.084    | 599       |
| 4           | True       | 0.080    | **625**   |
| 8           | False      | 0.150    | 334       |

Measured on M4 MacBook Air with a synthetic dataset. With real disk I/O the gap between `num_workers=0` and an optimal setting is typically 5–10×, not 2×.

Run it yourself:
```bash
python utils/dataloader_utils.py
```

### 3. Fine-tuning strategies with layer-wise learning rate decay

Three strategies for fine-tuning a pretrained model on a small dataset:

| Strategy     | Trainable params | When to use                                |
|--------------|------------------|--------------------------------------------|
| `head_only`  | 0.1% (~20K)      | Very small dataset (<1000 images)          |
| `last_block` | 63.7% (~15M)     | Medium dataset, partial adaptation         |
| `full`       | 100% (~24M)      | Larger dataset, full adaptation with LR decay |

`full` uses layer-wise learning rate decay:
layer1 (general features)   →  lr = 1e-5
layer2                      →  lr = 5e-5
layer3                      →  lr = 1e-4
layer4 (task-specific)      →  lr = 5e-4
head   (random init)        →  lr = 1e-3

The intuition: earlier layers learned general visual features during ImageNet pretraining. Updating them aggressively destroys what made the model useful. Later layers are more task-specific and can adapt faster. The head is random and needs the highest LR.

See `models/finetune.py`.

### 4. Custom losses without breaking the computational graph

The most common bug in custom losses: using NumPy operations on tensors. The moment a tensor passes through NumPy, it's detached from the computational graph. Gradients become `None` and weights never update. Verify with `loss.requires_grad` — it must be `True`.

This repo includes Dice Loss, Weighted Cross Entropy, and a combined BCE+Dice loss — all pure PyTorch, all graph-verified. See `utils/losses.py`.

### 5. NaN debugging — the three most common causes

When loss becomes NaN, the cause is almost always one of three things:

1. **NaN/Inf in your input data** — missing values, corrupted images, division by zero in preprocessing
2. **Bad loss function + input combination** — `log()` of zero, `BCELoss` with values outside `[0,1]`, label indices out of range
3. **Learning rate too high** — weights explode on the first update

The `NaNGuard` class wraps a training step and catches each failure mode at the earliest possible point, with messages that tell you which case fired and what to check.

```python
guard = NaNGuard(model, criterion, device)
for inputs, targets in loader:
    loss, grad_norm = guard.step(inputs, targets, optimizer)
```

See `utils/nan_debug.py`.

---

## Installation

```bash
git clone https://github.com/IslamFadl/pytorch-fundamentals.git
cd pytorch-fundamentals
pip install -r requirements.txt
```

Tested on Python 3.11, PyTorch 2.x, MacBook Air M-series (MPS backend) and CUDA.

---

## License

MIT — use freely, and please let me know if you found any issues while running the scripts on your machine.