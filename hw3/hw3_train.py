from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import random
import glob
import csv
import sys
import os

def load_data(img_path, label_path):
    train_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    train_label = pd.read_csv(label_path)
    train_label = train_label.iloc[:, 1].values.tolist()
    
    train_data = list(zip(train_image, train_label))
    random.shuffle(train_data)
    
    return train_data

def load_test_data(img_path):
    test_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    test_data = list(test_image)
    
    return test_data

class hw3_dataset(Dataset):
    def __init__(self, data, trans):
        self.data = data
        self.trans = trans
    
    def __getitem__(self, index):
        img = Image.open(self.data[index][0])
        img = self.trans(img)
        label = self.data[index][1]
        return img, label
    
    def __len__(self):
        return len(self.data)
    
class hw3_test_dataset(Dataset):
    def __init__(self, data, trans):
        self.data = data
        self.trans = trans
    
    def __getitem__(self, index):
        img = Image.open(self.data[index])
        img = self.trans(img)
        return img
    
    def __len__(self):
        return len(self.data)

class ConvNet(nn.Module):
    def __init__(self, num_type):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(3*3*128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_type))
    
    def forward(self, x):
        x = self.layer1(x) # 24*24
        x = self.layer2(x) # 12*12
        x = self.layer3(x) # 6*6
        x = self.layer4(x) # 3*3 with 128 layers
        x = x.view(-1, 3*3*128)
        result = self.fc(x)
        
        return result

Epoch = 100
num_type = 7

train_set = load_data(str(sys.argv[1]), str(sys.argv[2]))
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = hw3_dataset(train_set, transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

model = ConvNet(num_type)

use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()
    
crite = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
total_step = len(train_loader)

for epoch in range(Epoch):
    model.train()
    
    for batch_ind, (img, label) in enumerate(train_loader):
        if use_cuda:
            img = img.cuda()
            label = label.cuda()
            
        output = model(img)
        optimizer.zero_grad()
        loss = crite(output, label)
        loss.backward()
        optimizer.step()
                  
    model.eval()
    with torch.no_grad():
        train_loss = []
        train_acc = []
        for idx, (img, label) in enumerate(train_loader):
            if use_cuda:
                img = img.cuda()
                label = label.cuda()
            output = model(img)
            loss = crite(output, label)
            predict = torch.max(output, 1)[1]
            acc = np.mean((label == predict).cpu().numpy())

            train_acc.append(acc)
            train_loss.append(loss.item())
#         print("Epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))

    if (epoch+1) % 10 == 0:
        print(epoch+1)

torch.save(model.state_dict(), './model.pth')
