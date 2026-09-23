import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from pathlib import Path

class FER2013Conv(nn.Module):
    def __init__(self):

        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.Tanh(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.MaxPool2d(2),
            nn.Tanh()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3)),
            nn.Tanh(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.MaxPool2d(2),
            nn.Tanh()
        )

        self.dense_block = nn.Sequential(
            nn.Linear(5184, 1024),
            nn.Tanh(),
            nn.Linear(1024, 32),
            nn.Tanh(),
            nn.Linear(32, 7)
        )

    def forward(self, x):

        for i in self.conv_block1:
            x = i(x)

        for i in self.conv_block2:
            x = i(x)

        x = nn.Flatten()(x)

        for i in self.dense_block:
            x = i(x)

        return x

def train_one_epoch(model, loader, optimizer, criterion, epoch, epochs):
    
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

    for batch_idx, (x, y) in enumerate(pbar):
        optimizer.zero_grad()

        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x)
            loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{running_loss / (batch_idx + 1):.6f}"})

    return running_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x)
                val_loss += criterion(pred, y).item()

    return val_loss / len(loader)


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler,
                epochs, checkpoint_dir="checkpoints", label = 'best_model'):
    history = []

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val = np.inf

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch, epochs)
        scheduler.step()

        torch.cuda.empty_cache()

        val_loss = evaluate(model, val_loader, criterion)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, f"{checkpoint_dir}/{label}.pt")

        print(
            f"Epoch {epoch+1:03d} | "
            f"train = {train_loss:.6f} | "
            f"val = {val_loss:.6f} | "
            f"best_val = {best_val:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train": train_loss,
            "val": val_loss,
            "best_val": best_val
        })
    
    return history