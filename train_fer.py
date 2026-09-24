from modules import * 
from datasets import *
import torch

from torch.utils.data import DataLoader

def main():
    EPOCHS = 30
    BATCH_SIZE = 16
    PATH = '/home/lucaspaulogoncalves/data'
    OUTPUT_DIR = '/home/lucaspaulogoncalves/resultados'
    LABEL = 'fer2013'
    LR = 3e-3
    STEP_SIZE = 5

    train_ds = FER2013Dataset(PATH, 'train')
    test_ds = FER2013Dataset(PATH, 'test')

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=6, persistent_workers=True)
    val_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=3, persistent_workers=False)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    model = FER2013Conv()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, STEP_SIZE)
    loss = torch.nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, optimizer, loss, EPOCHS, scheduler, OUTPUT_DIR, LABEL)

if __name__ == '__main__':
    main()