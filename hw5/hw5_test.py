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
        
        # for weight in self.lstm.parameters():
        #     if len(weight.size()) > 1:
        #         weigth_init.orthogonal(weight.data)
        
    def forward(self, x, x_lens):
        x = self.embedd(x)
        output, hidden = self.lstm(x)
        
        x = output[:, -1, :]
        x = x.view(-1, 128)
        x = self.fc(x)
        
        return x

''' load data '''
print('...load data...')
test_x = me.load_data(str(sys.argv[1]))

''' take out words'''
print('...take out words...')
test_x = [i[1] for i in test_x]

''' word segmentation '''
print('...word segmentation...')
nlp = spacy.load("en_core_web_sm")
tokenizer = nlp.Defaults.create_tokenizer(nlp)
seg_test = me.word_segmentation(test_x, tokenizer)

''' load stop words and remove stop words '''
print('...load stop words and remove stop words...')
rmseg_test = me.remove_stopwords('./stop_words.txt', seg_test)

''' load dictionary '''
print('...load dictionary...')
with open ('diction_unique', 'rb') as fp:
    diction = pickle.load(fp)
fp.close()

''' load embedding model'''
print('...load embedding model...')
wv = me.load_word2vec('word2vec.model')

''' wv from numpy to tensor '''
print('...wv from numpy to tensor...')
weigh = np.array([])
for (keys, values) in diction.items():
    if values == 0:
        weigh = wv[str(keys)].reshape(1, -1)
    else:
        weigh = np.concatenate((weigh, wv[str(keys)].reshape(1, -1)), axis=0)

''' test preprocessing '''
print('...test preprocessing...')
test_dataset = Data(rmseg_test, mode='test')
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=me.test_pad_collate)

''' load model '''
print('...load model...')
model = LSTM()
model.load_state_dict(torch.load('./model.pth'))
model.eval()

use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()

''' predict '''
print('...predict...')
pred = me.predict(model, test_loader)

''' create csv '''
print('...create csv...')
df = pd.DataFrame({'id': np.arange(0, len(pred)), 'label': pred})
df.to_csv(str(sys.argv[2]), index=False)
