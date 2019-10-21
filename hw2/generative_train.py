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

def find_generative_model(x, y):
    type1 = x[np.argwhere(y == 1)][:, 0]
    type2 = x[np.argwhere(y == 0)][:, 0]
    
    N1 = type1.shape[0]
    N2 = type2.shape[0]
    mu1 = np.mean(type1, axis=0)
    mu2 = np.mean(type2, axis=0)
    covmat = N1/(N1+N2)*(np.cov(type1.T)) + N2/(N1+N2)*(np.cov(type2.T))
    
    w = np.dot((mu1-mu2).T, np.linalg.inv(covmat))
    b = (-1)/2*np.dot(np.dot(mu1.T, np.linalg.inv(covmat)), mu1) + 1/2*np.dot(np.dot(mu2.T, np.linalg.inv(covmat)), mu2) + np.log(N1/N2)
    
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

    w, b = find_generative_model(x_train, y_train)
    np.savez('generative_wb.npz', w=w, b=b)
