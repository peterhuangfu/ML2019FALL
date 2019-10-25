import numpy as np
import pandas as pd
import sys
import csv

def enhance(data):
    data = np.delete(data, np.s_[64:], axis=1)
    # data = np.delete(data, np.s_[15:22], axis=1)
    # data = np.delete(data, np.s_[6:15], axis=1)
    # data = np.delete(data, np.s_[3, 4], axis=1)

    return data

def sigmoid(z):
    prob = np.clip(1/(1 + np.exp(-z)), 1e-6, 1-1e-6)
    return prob

def find_logistic_model(x, y):
    n = x.shape[1]
    w = np.zeros(n)
    b = 0.0
    
    lr = 5*1e-3
    lr_w = np.ones(n)
    lr_b = 0.0
    iteration = 4000

    # logistic regression with gradient descent
    
    for ir in range(iteration):
        loss = []
        z = np.dot(x, w) + b
        
        lr_w = np.zeros(n)
        lr_b = 0.0
        z_sig = sigmoid(z)

        w_grad = np.dot((z_sig - y), x)
        b_grad = np.sum(z_sig - y)
        
        lr_w += w_grad**2
        lr_b += b_grad**2

        w -= lr/np.sqrt(lr_w) * w_grad
        b -= lr/np.sqrt(lr_b) * b_grad
        
        # if (ir+1) % 500 == 0:
        #     print(-1*np.sum(y*np.log(z_sig+1e-100) + (1-y)*np.log(1-z_sig+1e-100)))
        
    return w, b

if __name__ == '__main__':

    # load data

    X_train = pd.read_csv(str(sys.argv[3]))
    Y_train = pd.read_csv(str(sys.argv[4]), header=None)
    X_test = pd.read_csv(str(sys.argv[5]))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

    # standardize data

    num_index = [0, 1, 3, 4, 5]
    num_mu = np.zeros(X_train.shape[1])
    num_std = np.ones(X_train.shape[1])

    x_all = np.concatenate((X_train, X_test), axis=0)
    mu = np.mean(x_all, axis=0)
    std = np.std(x_all, axis=0)

    num_mu[num_index] = mu[num_index]
    num_std[num_index] = std[num_index]

    x_all_normal = (x_all - num_mu) / num_std
    x_train = x_all_normal[:X_train.shape[0]]
    x_test = x_all_normal[X_train.shape[0]:]
            
    # feature selection

    x_train = enhance(x_train)
    x_test = enhance(x_test)

    y_train = Y_train.reshape(-1, )

    # train

    w, b = find_logistic_model(x_train, y_train)
    np.savez('logistic_wb.npz', w=w, b=b)
