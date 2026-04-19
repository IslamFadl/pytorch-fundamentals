import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """
    A clean, reusable training loop that demonstrates correct PyTorch practice.

    Key concepts:
      - model.train() before training loop: enables Dropout + BatchNorm
        training behaviour (BatchNorm uses batch statistics, Dropout active)
      - model.eval() before validation loop: switches BatchNorm to use
        running statistics accumulated during training. Disables Dropout.
        FORGETTING THIS IS THE #1 MOST COMMON PYTORCH BUG.
      - torch.no_grad() during validation: disables gradient computation.
        Saves memory and speeds up inference.
        NOTE: this is DIFFERENT from model.eval(). You need BOTH.
      - optimizer.zero_grad() before backward: clears gradients from
        previous step. Forgetting this accumulates gradients incorrectly.
      - gradient clipping before optimizer.step(): prevents exploding
        gradients in deep networks and transformers.
    """

    def __init__(self, model, optimizer, criterion, device,
                 max_grad_norm=1.0, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train_epoch(self, loader):
        # ← CRITICAL: must call before every training loop
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="Train", leave=False)):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 1. Clear gradients from previous step
            self.optimizer.zero_grad()

            # 2. Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 3. Backward pass
            loss.backward()

            # 4. Gradient clipping — prevents NaN loss in deep/transformer models
            #    Caps gradient norm without changing gradient direction
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            # 5. Update weights
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def validate_epoch(self, loader):
        # ← CRITICAL: switches BatchNorm + Dropout to eval behaviour
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # ← CRITICAL: disables gradient computation (different from eval!)
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Val", leave=False):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(self, train_loader, val_loader, epochs):
        print(f"\nTraining on {self.device} for {epochs} epochs\n")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
                f"Val loss: {val_loss:.4f} acc: {val_acc:.3f}"
            )

        print("\nTraining complete.")
        return self.history