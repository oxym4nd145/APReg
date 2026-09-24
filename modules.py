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

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()
    curr_loss = 0

    batch_idx = 1
    batches = len(loader)

    for (x, y) in loader:
        optimizer.zero_grad()

        x = x.cuda()
        y = y.cuda()

        pred = model(x)
        error = criterion(pred, y)

        error.backward()
        optimizer.step()

        curr_loss += error.item()

        print(f"\rTrain batch :", batch_idx, '/', batches)
        batch_idx += 1

    return curr_loss/len(loader)

def eval(model, loader, optimizer, criterion, is_class):

    model.eval()
    curr_loss = 0
    correct = 0
    total = 0

    batch_idx = 1
    batches = len(loader)

    with torch.no_grad():
        for (x, y) in loader:

            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            error = criterion(pred, y)

            curr_loss += error.item()

            print(f"\rEval batch :", batch_idx, '/', batches)
            batch_idx += 1

            preds = pred.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(preds)

        return curr_loss/batches, correct/total


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, scheduler, output_dir, label):

    history = {
        'train_hist': [],
        'val_hist': [],
        'best_train_hist': [],
        'best_val_hist': []
    }

    best_train = np.inf
    best_val = np.inf

    for i in range(epochs):

        epoch_train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        torch.cuda.empty_cache()
        
        epoch_val_loss, epoch_acc = eval(model, val_loader, optimizer, criterion)

        if epoch_train_loss < best_train:
            best_train = epoch_train_loss

        if epoch_val_loss < best_val:
            best_val = epoch_val_loss

            torch.save({
                "epoch": i,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, f"{output_dir}/{label}.pt")

        print(f"Epoch {i+1}/{epochs}    |   Train error: {epoch_train_loss} Val error: {epoch_val_loss} Accuracy: {epoch_acc} \n|   Best train loss: {best_train}   Best val loss: {best_val}")

        history['train_hist'].append(epoch_train_loss)
        history['val_hist'].append(epoch_val_loss)
        history['best_train_hist'].append(best_train)
        history['best_val_hist'].append(best_val)

    return history