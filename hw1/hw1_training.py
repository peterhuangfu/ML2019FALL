import numpy as np
import pandas as pd
import re
import math
import csv

def read_data(data):
    # delete strange chars
    for col in list(data.columns):
        data[col] = data[col].astype(str).map(lambda x: x.rstrip('#x*A'))
    data = data.to_numpy()
    
    # convert special char to 0
    data[data == 'NR'] = 0
    data[data == ''] = 0
    data[data == 'nan'] = 0
    data = data.astype(np.float)
    
    return data

def extract_data(data):
    n = data.shape[0] // 18
    extract = data[:18, :]
    
    for i in range(1, n):
        extract = np.hstack((extract, data[i*18: i*18+18, :]))
    return extract

def if_valid(x, x_mean, x_std, y, y_mean, y_std):
    if y < 0 or y > (y_mean + 3.8*y_std):
        return False
    
    for i in range(9):
        if x[9, i] < 0 or x[9, i] > (x_mean + 3.8*x_std):
            return False
        
    return True

def parse_to_train(data):
    x = []
    y = []
    
    total_num = data.shape[1] - 9
    
    y_all = data[9, 9:]
    y_mean = np.mean(y_all)
    y_std = np.std(y_all)
    
    x_all = data[9, :total_num+8]
    x_mean = np.mean(x_all)
    x_std = np.mean(x_all)
    
    x_pm10_all = data[8, :total_num+8]
    x_pm10_mean = np.mean(x_pm10_all)
    x_pm10_std = np.std(x_pm10_all)
    x_pm10_not_outlier_mean = np.mean(x_pm10_all[np.logical_and(x_pm10_all >= 0, x_pm10_all <= x_pm10_mean + 3.8*x_pm10_std)])

    for i in range(total_num+8):
        if data[8, i] < 0 or data[8, i] > (x_pm10_mean + 3.8*x_pm10_std):
            data[8, i] = x_pm10_not_outlier_mean
    
    for i in range(total_num):
        x_temp = data[:, i: i+9]
        y_temp = data[9, i+9]
        
        if if_valid(x_temp, x_mean, x_std, y_temp, y_mean, y_std):
            x.append(np.append(x_temp.reshape(-1, ), 1))
            y.append(y_temp)
            
    x = np.array(x)
    y = np.array(y)
    
    return x, y

def training(train_x, train_y):
    lr = 0.1
    lr_w = 0.0
    iteration = 2000

    rmse = 0.0
    n = train_x.shape[1]
    w = np.zeros(n)

    # linear regression with gradient descent
    for ir in range(iteration):
        rmse = 0.0
        loss = []
        for i in range(len(train_x)):
            lo = np.dot(train_x[i], w.T) - train_y[i]
            loss.append(lo)
            rmse += lo ** 2

        w_grad = 2*np.dot(train_x.T, np.array(loss))
        rmse /= len(train_y)
        lr_w += w_grad**2

        w -= lr/np.sqrt(lr_w) * w_grad
        
#         if (ir+1) % 100 == 0:
#             print(ir+1)
#             print('rmse = ', np.sqrt(rmse))
        
    return w

year1 = pd.read_csv('./year1.csv')
year2 = pd.read_csv('./year2.csv')

year1 = year1.drop(columns=['日期','測項'])
year2 = year2.drop(columns=['日期','測項'])

train = pd.concat([year1, year2], axis=0)

train_data = read_data(train)
extracted_data = extract_data(train_data)
train_x, train_y = parse_to_train(extracted_data)

w = training(train_x, train_y)
np.save('weight', w)
