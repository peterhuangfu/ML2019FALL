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

def load_test_data(img_path):
    test_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    test_data = list(test_image)
    
    return test_data

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

num_type = 7
test_set = load_test_data(str(sys.argv[1]))
transform = transforms.Compose([transforms.ToTensor()])

test_dataset = hw3_test_dataset(test_set, transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

test_model = ConvNet(num_type)
test_model.load_state_dict(torch.load('./model.pth'))
test_model.eval()

use_cuda = torch.cuda.is_available()
if use_cuda:
    test_model.cuda()

pred = []
with torch.no_grad():
    for batch_ind, img in enumerate(test_loader):
        if use_cuda:
            img = img.cuda()
        res = test_model(img)
        pred_label = torch.max(res, 1)[1]
        pred.append(pred_label)

label_pred = []
for batch in pred:
    for e in batch.data.cpu().numpy():
        label_pred.append(e)

with open(str(sys.argv[2]), 'w', newline='') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(['id', 'label'])
    for row_ind, row in enumerate(label_pred):
        wr.writerow([str(row_ind), row])
