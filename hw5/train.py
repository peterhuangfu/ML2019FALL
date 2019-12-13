import numpy as np
import pandas as pd
import csv
import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader, Dataset

def train_model(model, Epoch, train_loader, criterion, optimizer):
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        model.cuda()
    
    for epoch in range(Epoch):
        model.train()

        for idx, (sentence, label, x_lens, y_lens) in enumerate(train_loader):
            if use_gpu:
                sentence = sentence.cuda()
                label = label.cuda()

            output = model(sentence, x_lens)

            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = []
            train_acc = []
            for idx, (sentence, label, x_lens, y_lens) in enumerate(train_loader):
                if use_gpu:
                    sentence = sentence.cuda()
                    label = label.cuda()

                output = model(sentence, x_lens)

                loss = criterion(output, label)
                predict = torch.max(output, 1)[1]
                acc = np.mean((label == predict).cpu().numpy())

                train_acc.append(acc)
                train_loss.append(loss.item())

        # if (epoch+1) % 10 == 0:
        # print("Epoch: {}, train_loss: {:.5f}, train_acc: {:.5f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
        
    return model

def valid_model(model, Epoch, subtrain_loader, subtest_loader, criterion, optimizer):
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        model.cuda()
    
    for epoch in range(Epoch):
        model.train()

        for idx, (sentence, label, x_lens, y_lens) in enumerate(subtrain_loader):
            if use_gpu:
                sentence = sentence.cuda()
                label = label.cuda()

            output = model(sentence, x_lens)

            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = []
            train_acc = []
            for idx, (sentence, label, x_lens, y_lens) in enumerate(subtest_loader):
                if use_gpu:
                    sentence = sentence.cuda()
                    label = label.cuda()

                output = model(sentence, x_lens)

                loss = criterion(output, label)
                predict = torch.max(output, 1)[1]
                acc = np.mean((label == predict).cpu().numpy())

                train_acc.append(acc)
                train_loss.append(loss.item())

        # if (epoch+1) % 10 == 0:
        # print("Epoch: {}, train_loss: {:.5f}, train_acc: {:.5f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
        if (np.mean(train_acc) > 0.718):
            break
    return model
