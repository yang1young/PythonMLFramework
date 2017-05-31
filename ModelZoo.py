#!/usr/bin/python
# coding=utf-8
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_recall_curve,precision_score,recall_score
import datetime
import pandas as pd
from sklearn.tree import export_graphviz
import numpy as np
import sklearn.svm.libsvm as libsvm
from sklearn import linear_model


class models():

    model = None

    def __init__(self,name):
        self.name = name

    def _build(self,x_train,y_train,path):
        fp = open(path + 'log.txt', 'a')
        fp.write('Start training '+self.name + '\n')
        start = datetime.datetime.now()
        self.model.fit(x_train, y_train)
        end = datetime.datetime.now()
        print self.name+' training time is '+str(end - start)
        fp.write(self.name+' training time is '+str(end - start) + '\n')
        fp.close()

    def modelEvaluate(self, x_test,y_test,path):
        start = datetime.datetime.now()
        pred_y = self.model.predict(x_test)
        end = datetime.datetime.now()
        fp = open(path + 'log.txt', 'a')
        fp.write('Start evaluate '+self.name + '\n')
        fp.write(self.name + ' predicte time is ' + str(end - start) + '\n')
        print self.name + ' predicte time is ' + str(end - start)

        print 'crosstab:{0}'.format(pd.crosstab(y_test, pred_y))
        print 'accuracy_score:{0}'.format(accuracy_score(y_test, pred_y))
        print '0precision_score:{0}'.format(precision_score(y_test, pred_y, pos_label=0))
        print '0recall_score:{0}'.format(recall_score(y_test, pred_y, pos_label=0))
        print '1precision_score:{0}'.format(precision_score(y_test, pred_y, pos_label=1))
        print '1recall_score:{0}'.format(recall_score(y_test, pred_y, pos_label=1))

        fp.write('crosstab:{0}'.format(pd.crosstab(y_test, pred_y))+ '\n')
        fp.write('accuracy_score:{0}'.format(accuracy_score(y_test, pred_y)) + '\n')
        fp.write('0precision_score:{0}'.format(precision_score(y_test, pred_y, pos_label=0)) + '\n')
        fp.write('0recall_score:{0}'.format(recall_score(y_test, pred_y, pos_label=0)) + '\n')
        fp.write('1precision_score:{0}'.format(precision_score(y_test, pred_y, pos_label=1)) + '\n')
        fp.write('1recall_score:{0}'.format(recall_score(y_test, pred_y, pos_label=1)) + '\n')

        fp.close()

    def predictP(self,input):
        input = np.array(input).reshape(1, -1)
        result = self.model.predict_proba(input)
        return result

    def predict(self,input):
        input = np.array(input)
        result = self.model.predict(input)
        return result

    def top_probality(self,x_test,path,top_num):
        predication_prob = self.model.predict_proba(x_test)[:, 1]
        index = np.argsort(predication_prob)[-top_num:][::-1]
        return index

class DecisionTree(models):

    def build(self, x_train,y_train,path,**parameter):
        self.model = DecisionTreeClassifier(**parameter)
        self._build(x_train,y_train,path)

    def saveTree(self,path):
        with open(path+"dt.dot", 'w') as f:
            export_graphviz(self.model, out_file=f,feature_names=
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14'])


class GBDT(models):
    def build(self, x_train, y_train, path,**parameter):
        self.model =GradientBoostingClassifier(**parameter)
        self._build(x_train, y_train,path)


class RF(models):
    def build(self, x_train, y_train, path,**parameter):
        self.model = RandomForestClassifier(**parameter)
        self._build(x_train, y_train,path)

    def featureImportance(self,x_train,path):
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        #print indices
        col_list = x_train.columns
        fp = open(path + 'log.txt', 'a')
        for f, name,_ in zip(range(x_train.shape[1]), col_list,range(50)):
            print "%d. feature %d (%f)(name = %s)" % (f + 1, indices[f], importance[indices[f]], col_list[indices[f]])
            fp.write("%d. feature %d (%f)(name = %s)\n" % (f + 1, indices[f], importance[indices[f]], col_list[indices[f]]))
        fp.close()


class SVM(models):

    def build(self, x_train, y_train,path, **parameter):
        x = x_train.as_matrix()
        x = x.copy(order='C').astype(np.float64)
        y = y_train.as_matrix().astype(np.float64)

        self.model = libsvm.fit(x, y, **parameter)

    def predictP(self, input):
        input = np.array(input).reshape(1, -1)
        result = libsvm.predict_proba(input,*(self.model))
        return result

    def predict(self, input):
        input = np.array(input)
        result = libsvm.predict(input, *(self.model))
        return result

    def top_probality(self, x_test, path, top_num):
        predication_prob = libsvm.predict_proba(x_test,*(self.model))[:, 1]
        index = np.argsort(predication_prob)[-top_num:][::-1]
        print index

    def modelEvaluate(self, x_test, y_test,path):
        start = datetime.datetime.now()
        pred_y = libsvm.predict(x_test.as_matrix().copy(order='C').astype(np.float64), *(self.model))
        end = datetime.datetime.now()
        print self.name + ' predicte time is ' + str(end - start)
        print 'crosstab:{0}'.format(pd.crosstab(y_test, pred_y))
        print 'accuracy_score:{0}'.format(accuracy_score(y_test, pred_y))
        print '0precision_score:{0}'.format(precision_score(y_test, pred_y, pos_label=0))
        print '0recall_score:{0}'.format(recall_score(y_test, pred_y, pos_label=0))
        print '1precision_score:{0}'.format(precision_score(y_test, pred_y, pos_label=1))
        print '1recall_score:{0}'.format(recall_score(y_test, pred_y, pos_label=1))
        fp = open(path + 'log.txt', 'a')
        fp.write('crosstab:{0}'.format(pd.crosstab(y_test, pred_y)) + '\n')
        fp.write('accuracy_score:{0}'.format(accuracy_score(y_test, pred_y)) + '\n')
        fp.write('0precision_score:{0}'.format(precision_score(y_test, pred_y, pos_label=0)) + '\n')
        fp.write('0recall_score:{0}'.format(recall_score(y_test, pred_y, pos_label=0)) + '\n')
        fp.write('1precision_score:{0}'.format(precision_score(y_test, pred_y, pos_label=1)) + '\n')
        fp.write('1recall_score:{0}'.format(recall_score(y_test, pred_y, pos_label=1)) + '\n')
        fp.close()

class lasso(models):
    def build(self, x_train, y_train, path,**parameter):
        self.model =linear_model.LogisticRegression(**parameter)
        self._build(x_train, y_train,path)


