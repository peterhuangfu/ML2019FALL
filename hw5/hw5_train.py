import numpy as np
import pandas as pd
import sys
import csv
import torch
import spacy
import torch.nn as nn
import gensim as gs
import copy
import emoji
import torch.nn.init as weigth_init
import pickle
import train
import methods as me

from torch import optim
from torch.utils.data import DataLoader, Dataset

''' model class '''
class Data(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
    
    def __getitem__(self, index):
        if self.mode == 'train':
            sentence = self.data[index][0]
            label = self.data[index][1]
            sentence = [diction[str(e)] for e in sentence]
            
            return sentence, label
            # return me.pass_embedding(sentence, wv, 256), label
        else:
            sentence = self.data[index]
            sentence = [diction[str(e)] for e in sentence]
            
            return sentence
            # return me.pass_embedding(sentence, wv, 256)
    
    def __len__(self):
        return len(self.data)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.embedd = nn.Embedding(weigh.shape[0], 256)
        self.embedd.weight.data.copy_(torch.from_numpy(weigh))
        self.embedd.weight.requires_grad = False
        
        self.lstm = nn.LSTM(256, 128, num_layers=1, dropout=0.37, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2),
            nn.Sigmoid())
        
        for weight in self.lstm.parameters():
            if len(weight.size()) > 1:
                weigth_init.orthogonal(weight.data) 
        
    def forward(self, x, x_lens):
        x = self.embedd(x)
        output, hidden = self.lstm(x)
        
        x = output[:, -1, :]
        x = x.view(-1, 128)
        x = self.fc(x)
        
        return x

''' load data '''
print('...load data...')
train_x = me.load_data(str(sys.argv[1]))
train_y = me.load_data(str(sys.argv[2]))
test_x = me.load_data(str(sys.argv[3]))

''' take out words'''
print('...take out words...')
train_x = [i[1] for i in train_x]
train_y = [i[1] for i in train_y]
test_x = [i[1] for i in test_x]

''' word segmentation '''
print('...word segmentation...')
nlp = spacy.load("en_core_web_sm")
tokenizer = nlp.Defaults.create_tokenizer(nlp)
seg_train = me.word_segmentation(train_x, tokenizer)
seg_test = me.word_segmentation(test_x, tokenizer)

''' load stop words and remove stop words and get set '''
print('...load stop words and remove stop words and get set...')
rmseg_train = me.remove_stopwords('./stop_words.txt', seg_train)
rmseg_test = me.remove_stopwords('./stop_words.txt', seg_test)

''' concat train and test '''
print('...concat train and test...')
train_dict = copy.deepcopy(rmseg_train)
for i in rmseg_test:
    train_dict.append(i)

''' build index dictionary and save '''
print('...build index dictionary and save...')
diction = me.build_idx_dict(train_dict)
with open('diction_unique_test_submit', 'wb') as fp:
    pickle.dump(diction, fp)
fp.close()

''' word embedding with gensim.models.Word2Vec '''
print('...word embedding with gensim.models.Word2Vec...')
wv = me.build_word2vec(train_dict, 256, 1500, 'word2vec_test_submit.model')

''' wv from numpy to tensor '''
print('...wv from numpy to tensor...')
weigh = np.array([])
for (keys, values) in diction.items():
    if values == 0:
        weigh = wv[str(keys)].reshape(1, -1)
    else:
        weigh = np.concatenate((weigh, wv[str(keys)].reshape(1, -1)), axis=0)

''' train preprocessing '''
print('...train preprocessing...')
train_dataset = Data(list(zip(rmseg_train, train_y)), mode='train')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=me.pad_collate)

criterion = nn.CrossEntropyLoss()

''' validation '''
print('...validation...')
from sklearn.model_selection import train_test_split
x_subtrain, x_subtest, y_subtrain, y_subtest = train_test_split(rmseg_train, train_y, test_size=0.1, random_state=212)

subtrain_dataset = Data(list(zip(x_subtrain, y_subtrain)), mode='train')
subtest_dataset = Data(list(zip(x_subtest, y_subtest)), mode='train')

subtrain_loader = DataLoader(subtrain_dataset, batch_size=64, shuffle=True, collate_fn=me.pad_collate)
subtest_loader = DataLoader(subtest_dataset, batch_size=64, shuffle=True, collate_fn=me.pad_collate)

criterion = nn.CrossEntropyLoss()

Epoch = 50
submodel = LSTM()
valid_optimizer = optim.Adam(submodel.parameters(), lr=4.5*1e-4)
submodel = train.valid_model(submodel, Epoch, subtrain_loader, subtest_loader, criterion, valid_optimizer)

''' train model '''
print('...train model...')
Epoch = 50
model = LSTM()
train_optimizer = optim.Adam(model.parameters(), lr=4.5*1e-4)
model = train.train_model(model, Epoch, train_loader, criterion, train_optimizer)

''' save model '''
print('...save model...')
torch.save(submodel.state_dict(), './model.pth')
