import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import model

from torch import optim
from torch.utils.data import DataLoader, Dataset

Epoch = 150
use_gpu = torch.cuda.is_available()

auto_encoder = model.AutoEncoder()

train_set = np.load(str(sys.argv[1]))
train_x = np.transpose(train_set, (0, 3, 1, 2)) / 255. * 2 - 1
train_x = torch.Tensor(train_x)

if use_gpu:
  auto_encoder.cuda()
  train_x = train_x.cuda()
    
train_loader = DataLoader(train_x, batch_size=60, shuffle=True)

criterion = nn.L1Loss()
optimizer = optim.Adam(auto_encoder.parameters(), lr=0.005)

for epoch in range(Epoch):
  accumu_loss = 0
  
  for x in train_loader:
    latent, reconstruct = auto_encoder(x)
    loss = criterion(reconstruct, x)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    accumu_loss += loss.item() * x.shape[0]
  
  if (epoch+1) % 10 == 0:
    print(f'Epoch: {epoch + 1}  Loss: {"%.5f" % (accumu_loss / train_x.shape[0])}')

torch.save(auto_encoder.state_dict(), './model.pth')
