import numpy as np
import pandas as pd
import re
import math
import csv
import sys

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

def test_valid(x, x_mean, x_std, x_correct_mean):
    for i in range(9):
        if x[9, i] < 0 or x[9, i] > (x_mean + 3.8*x_std):
            x[9, i] = x_correct_mean
    return x

def parse_test_data(data):
    x = []
    
    total_num = data.shape[1] // 9
    
    x_all = data[9, :]
    
    x_mean = np.mean(x_all)
    x_std = np.std(x_all)
    x_not_outlier = x_all[np.logical_and(x_all >= 0, x_all <= (x_mean + 3.8*x_std))]
    x_correct_mean = np.mean(x_not_outlier)
    
#     x_pm10_all = data[8, :total_num+8]
#     x_pm10_mean = np.mean(x_pm10_all)
#     x_pm10_std = np.std(x_pm10_all)
#     x_pm10_not_outlier_mean = np.mean(x_pm10_all[np.logical_and(x_pm10_all >= 0, x_pm10_all <= x_pm10_mean + 3.8*x_pm10_std)])
    
#     for i in range(total_num+8):
#         if data[8, i] < 0 or data[8, i] > (x_pm10_mean + 3.8*x_pm10_std):
#             data[8, i] = x_pm10_not_outlier_mean
            
    for row in range(18):
        if row == 0 or row == 9:
            continue
        else:
            x_other_all = data[row, :]
            x_other_mean = np.mean(x_other_all)
            x_other_std = np.std(x_other_all)
            x_other_not_outlier_mean = np.mean(x_other_all[np.logical_and(x_other_all >= 0, x_other_all <= (x_other_mean + 3.8*x_other_std))])
            
            for i in range(total_num):
                if data[row, i] < 0 or data[row, i] > (x_other_mean + 3.8*x_other_std):
                    data[row, i] = x_other_not_outlier_mean
    
    for i in range(total_num):
        x_temp = test_valid(data[:, i*9: i*9+9], x_mean, x_std, x_correct_mean)
        x.append(np.append(x_temp.reshape(-1, ), 1))
        
    x = np.array(x)
    
    return x

test = pd.read_csv(str(sys.argv[1]))
test = test.drop(columns=['id','測項'])

w = np.load('weight.npy')

test_data = read_data(test)
extracted_test = extract_data(test_data)
test_x = parse_test_data(extracted_test)

test_y = np.dot(test_x, w)

with open(str(sys.argv[2]), 'w', newline='') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(['id', 'value'])
    for row_ind, row in enumerate(test_y):
        wr.writerow(['id_'+str(row_ind), row])
