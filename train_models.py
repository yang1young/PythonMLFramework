#!/usr/bin/python
# coding=utf-8
import model_zoo
import numpy as np
import ml_utils as ml
import data_prepare as dp

MODEL_PATH = '/home/yangqiao/pythonProject/PythonMLFramework/model/'
LOG_PATH = '/home/yangqiao/pythonProject/PythonMLFramework/log/'


# train decision tree
def trainDT(x_train, x_test, y_train, y_test, path):
    dt = model_zoo.DecisionTree('dt')
    dt.build(x_train, y_train, path, max_depth=10)
    dt.modelEvaluate(x_test, y_test, path)
    dt.saveTree(path)
    ml.modelDump(dt, path + 'dt.txt')
    return dt


# train svm model
def trainSVM(x_train, x_test, y_train, y_test, path):
    svm = model_zoo.SVM('SVM')
    svm.build(x_train, y_train, path, C=10.0, kernel='rbf', class_weight=np.asarray([1.0, 10.0]), gamma=1,
              shrinking=True, probability=True)
    svm.modelEvaluate(x_test, y_test, path)
    ml.modelDump(svm, path + "SVM.txt")
    return svm


# train GBDT
def trainGBDT(x_train, x_test, y_train, y_test, path):
    GBDT = model_zoo.GBDT('GBDT')
    GBDT.build(x_train, y_train, path, subsample=0.7)
    GBDT.modelEvaluate(x_test, y_test, path)
    ml.modelDump(GBDT, path + "gbdt.txt")
    return GBDT


# train random forest
def trainRF(x_train, x_test, y_train, y_test, path):
    rf = model_zoo.RF('RF')
    rf.build(x_train, y_train, path, n_estimators=200, max_depth=6)
    rf.modelEvaluate(x_test, y_test, path)
    rf.featureImportance(x_train, path)
    ml.modelDump(rf, path + "rf.txt")
    return rf


# train lasso
def trainLasso(x_train, x_test, y_train, y_test, path):
    lasso = model_zoo.lasso('lasso')
    lasso.build(x_train, y_train, path, penalty='l1', C=1, class_weight={0: 1, 1: 20}, max_iter=100)
    lasso.modelEvaluate(x_test, y_test, path)
    # lasso.top_probality(x_test,path)
    ml.modelDump(lasso, path + "lasso.txt")
    return lasso


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = dp.data_prepare(LOG_PATH, LOG_PATH, 0, 20)
    # trainRF(x_train, x_test, y_train, y_test,MODEL_PATH)
    # trainLasso(x_train, x_test, y_train, y_test,MODEL_PATH)
    # trainGBDT(x_train, x_test, y_train, y_test,MODEL_PATH)
    # trainDT(x_train, x_test, y_train, y_test,MODEL_PATH)
    ml.plotMargin(LOG_PATH, 0, 1)
