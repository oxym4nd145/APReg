from modules import *
from datasets import *

import torch
from torch.utils.data import DataLoader

OUTPUT_DIR = '/home/lucaspaulogoncalves/resultados'
PATH = '/home/lucaspaulogoncalves/data'
BATCH_SIZE = 16
SEED = 42
EPOCHS = 100
LR = 1e-3

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
print(f"Torch: {torch.__version__}  |  Torchvision: {__import__('torchvision').__version__}")

train_ds = FER2013Dataset(PATH, 'train')
val_ds = FER2013Dataset(PATH, 'test')

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=3,
    pin_memory=True,
    persistent_workers=False,
)

model = FER2013Conv()
print(model.parameters())

optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=LR, 
    steps_per_epoch=len(train_loader), 
    epochs=EPOCHS,
    pct_start=0.2,
    anneal_strategy='cos'
)

criterion = nn.CrossEntropyLoss()

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    epochs=EPOCHS,
    checkpoint_dir=OUTPUT_DIR,
    label='teste_fer'
)