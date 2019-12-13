import numpy as np
import pandas as pd
import csv
import torch
import spacy
import torch.nn as nn
import gensim as gs
import copy
import emoji

from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def load_data(path):
    data = pd.read_csv(path).values.tolist()
    
    return data

def word_segmentation(data, tokenizer):
    seg = [[token.orth_ for token in tokenizer(senten)] for senten in data]
    
    return seg

def remove_stopwords(stop_path, data):
    stops = list()
    with open(stop_path, 'r') as stop:
        for w in stop.readlines():
            w = w.split('\n')
            stops.append(w[0])
    stop.close()

    rmseg_data = [None]*len(data)

    for idx, senten in enumerate(data):
        for widx, word in enumerate(senten):
            if word not in stops and word not in emoji.UNICODE_EMOJI:
                if rmseg_data[idx] == None:
                    rmseg_data[idx] = []
                    rmseg_data[idx].append(word)
                else:
                    rmseg_data[idx].append(word)
        if rmseg_data[idx] == None:
            rmseg_data[idx] = []
            
    return rmseg_data 
    
def build_idx_dict(data):
    diction = dict()
    dict_set = []
    for each in data:
        for e in each:
            dict_set.append(e)
    dict_set = list(set(dict_set))
    for idx, i in enumerate(dict_set):
        diction[i] = idx

    return diction

def build_word2vec(data, size, epoch, save_path):
    embedd_model = gs.models.Word2Vec(data, size=size, min_count=1, workers=4)
    embedd_model.train(data, total_examples=len(data), epochs=epoch)
    embedd_model.save(save_path)
    wv = embedd_model.wv
    
    return wv

def load_word2vec(path):
    embedd_model = gs.models.Word2Vec.load(path)
    wv = embedd_model.wv
    
    return wv

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    
    x_lens = [len(x) for x in xx]
    y_lens = len(list(yy))
    
    xx = [torch.tensor(x) for x in xx]
    yy = torch.tensor(yy)
    
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

    return xx_pad, yy, x_lens, y_lens

def test_pad_collate(batch):
    xx = batch
    x_lens = [len(x) for x in xx]
    xx = [torch.tensor(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

    return xx_pad, x_lens

def pass_embedding(sentence, wv, size):
    embedded = np.zeros(size)
    for ind, word in enumerate(sentence):
        if ind == 0:
            embedded = wv[str(word)]
            embedded = embedded.reshape(-1, size)
        else:
            emb = wv[str(word)]
            emb = emb.reshape(-1, size)
            embedded = np.concatenate((embedded, emb), axis=0)
            
    return embedded

def predict(model, test_loader):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
        
    pred = []
    with torch.no_grad():
        for idx, (sentence, x_lens) in enumerate(test_loader):
            if use_gpu:
                sentence = sentence.cuda()

            output = model(sentence, x_lens)
            pred_label = torch.max(output, 1)[1]
            pred.append(pred_label)

    label_pred = []
    for batch in pred:
        for e in batch.data.cpu().numpy():
            label_pred.append(e)

    return label_pred
