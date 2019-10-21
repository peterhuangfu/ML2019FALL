import numpy as np
import pandas as pd
import sys
import csv

from sklearn.ensemble import GradientBoostingClassifier as gbc

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

    # initialize model and test and output result

    gbc_best = gbc(n_estimators=350, learning_rate=0.15, random_state=112).fit(x_train, y_train)
    y_predict = gbc_best.predict(x_test)

    with open(str(sys.argv[6]), 'w', newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(['id', 'label'])
        for row_ind, row in enumerate(y_predict):
            wr.writerow([str(row_ind+1), row])
