import numpy as np
import pandas as pd
import sys
import csv

def enhance(data):
    data = np.delete(data, np.s_[64:], axis=1)
    data = np.delete(data, np.s_[15:22], axis=1)
    # data = np.delete(data, np.s_[6:15], axis=1)
    data = np.delete(data, np.s_[3, 4], axis=1)

    return data

def sigmoid(z):
    prob = np.clip(1/(1 + np.exp(-z)), 1e-6, 1-1e-6)
    return prob

if __name__ == '__main__':

    # load data

    X_train = pd.read_csv(str(sys.argv[3]))
    Y_train = pd.read_csv(str(sys.argv[4]), header=None)
    X_test = pd.read_csv(str(sys.argv[5]))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
            
    # feature selection

    x_train = enhance(X_train)
    x_test = enhance(X_test)

    y_train = Y_train.reshape(-1, )

    # load model

    model = np.load('./generative_wb.npz')
    w = model['w']
    b = model['b']

    # test and output result

    y_predict = []
    z_te = np.dot(x_test, w) + b
    pred_te = sigmoid(z_te)
    for i in pred_te:
        y_predict.append(1 if i >= 0.5 else 0)

    with open(str(sys.argv[6]), 'w', newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(['id', 'label'])
        for row_ind, row in enumerate(y_predict):
            wr.writerow([str(row_ind+1), row])
