import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    
    self.encoder1 = nn.Sequential(
      nn.Conv2d(3, 8, 3, 2, 1),
      nn.LeakyReLU(negative_slope=0.05),
      nn.BatchNorm2d(8),
      
      nn.Conv2d(8, 16, 3, 2, 1),
      nn.LeakyReLU(negative_slope=0.05),
      nn.BatchNorm2d(16),
      
      nn.Conv2d(16, 32, 3, 2, 1),
      nn.LeakyReLU(negative_slope=0.05),
      nn.BatchNorm2d(32))
    
    self.encoder2 = nn.Sequential(
      nn.Linear(4*4*32, 256),
      nn.ReLU(),
      nn.BatchNorm1d(256),
      nn.Linear(256, 128))
    
    self.decoder1 = nn.Sequential(
      nn.Linear(128, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Linear(256, 4*4*32))
    
    self.decoder2 = nn.Sequential(
      nn.ConvTranspose2d(32, 16, 2, 2),
      nn.ConvTranspose2d(16, 8, 2, 2),
      nn.ConvTranspose2d(8, 3, 2, 2),
      nn.Tanh())
      
  def forward(self, x):
    encoded1 = self.encoder1(x)
    encoded1 = encoded1.view(-1, 4*4*32)
    encoded = self.encoder2(encoded1)
    
    decoded1 = self.decoder1(encoded)
    decoded1 = decoded1.view(60, 32, 4, 4)
    decoded = self.decoder2(decoded1)
    
    return encoded, decoded
