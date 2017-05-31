# !/usr/bin/python
# coding=utf-8
import pickle
import TrainModels as M
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt


def modelDump(model,path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def modelReload(path):
    with open(path, 'rb') as f:
        model2 = pickle.load(f)
    return model2


def plotMargin(dir,index1,index2):
    csv = pd.read_csv(dir+'train.csv')
    csv.columns = range(0, len(csv.columns), 1)
    X = csv.iloc[:,[index1,index2]]
    Y = csv.iloc[:,-1]
    print X.shape

    RandomForest = M.trainRF(X,X,Y,Y, dir)
    SVM = M.trainSVM(X,X,Y,Y, dir)
    GBDT = M.trainGBDT(X,X,Y,Y, dir)
    DecisionTree = M.trainDT(X,X,Y,Y, dir)

    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    print xx.shape ,yy.shape
   # temp  = pd.concat([pd.DataFrame(xx).T,pd.DataFrame(yy).T],axis=1)
    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))
    for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [RandomForest,SVM, GBDT, DecisionTree],
                        ['RandomForest', 'SVM(RBF)',
                         'GBDT', 'DecisionTree']):
        temp  = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
        Z = clf.predict(temp)
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y, alpha=0.8)
        axarr[idx[0], idx[1]].set_title(tt)

    plt.show()

