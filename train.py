
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import AnomalyDataset, img_load
from models.get_models import Detector
from utils.evaluate import score_function

device = torch.device('cuda')

train_png = sorted(glob('data/train/*.png'))
train_imgs = [img_load(m) for m in tqdm(train_png)]

train_y = pd.read_csv('data/train_df.csv')
train_labels = train_y['label']

label_unique = sorted(np.unique(train_labels))
label_unique = {k: v for k, v in zip(label_unique, range(len(label_unique)))}

train_labels = [label_unique[k] for k in train_labels]

dataset = AnomalyDataset(np.array(train_imgs), np.array(train_labels), mode='train')

train_dataset = ''
val_dataset = ''

train_loader = DataLoader(train_dataset)
val_loader = DataLoader(val_dataset)

model = Detector().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()


for epoch in range(10):
    model.train()
    for batch in tqdm(train_loader):
        img = torch.tensor(batch[0], dtype=torch.float32, device=device)
        target = torch.tensor(batch[1], dtype=torch.long, device=device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = model(img)
        loss = criterion(pred, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        