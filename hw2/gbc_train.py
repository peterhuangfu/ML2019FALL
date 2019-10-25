import numpy as np
import pandas as pd
import sys
import csv

from sklearn.model_selection import cross_validate as cva
from sklearn.ensemble import GradientBoostingClassifier as gbc

# adopt cross validation to tune gbc models and find the best one

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

    y_train = Y_train.reshape(-1, )

    # tune model

    top_n = 0.0
    top_lr = 0.0
    top_score = 0.0

    # it takes 2.5 hours to tune the model via Intel i5 CPU core

    for n in [50, 80, 100, 150, 200, 250, 300, 350]:
        for lr in [0.001, 0.005, 0.01, 0.0375, 0.05, 0.0875, 0.1, 0.15]:
            gbc_model = gbc(n_estimators=n, learning_rate=lr, random_state=112)
            res = cva(gbc_model, x_train, y_train, cv=10, n_jobs=-1)
            final_score = np.mean(res['test_score'])
            if final_score > top_score:
                top_score = final_score
                top_n = n
                top_lr = lr
              
    print('top_n = ', top_n, ' top_lr = ', top_lr, ' top_score = ', top_score)
