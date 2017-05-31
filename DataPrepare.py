#!/usr/bin/python
# coding=utf-8
import pandas as pd
from sklearn.cross_validation import train_test_split

TRAIN_PATH='/home/yangqiao/pythonProject/Python_ML_Framework/wine.csv'
TEST_PATH='/home/yangqiao/pythonProject/Python_ML_Framework/wine.csv'
DELETE_INDEX = []
LABLE_INDEX = 0

def data_clean(data_file):
    csv = pd.read_csv(data_file)
    csv.columns = range(0, len(csv.columns), 1)
    if(LABLE_INDEX!=-1):
        Y = csv.iloc[:, LABLE_INDEX]
        X = csv.drop(LABLE_INDEX,axis=1)
    else:
        X = csv.iloc[:, :len(csv.columns.tolist()) - 1]
        Y = csv.iloc[:, -1]
    if(DELETE_INDEX != None):
        # delete some columns
        X = X.drop(DELETE_INDEX, axis=1)
    data = pd.concat([X,Y], axis=1)
    return data


def sub_sample(data_file,sub_sample_time):
    train_black = data_file[data_file.iloc[:, len(data_file.columns) - 1] == 1]
    train_white = data_file[data_file.iloc[:, len(data_file.columns) - 1] == 0]
    # sampletime
    percentage = float(train_black.shape[0] * sub_sample_time) / float(train_white.shape[0])
    train_white_sample = train_white.sample(frac=percentage, replace=False)
    train = pd.DataFrame(pd.concat([train_white_sample, train_black], axis=0))
    x_train = train.iloc[:, :len(train.columns.tolist()) - 1]
    y_train = train.iloc[:, -1]
    return x_train,y_train


def get_train_test_split(data_file,split_percentage):

    x_train, x_test, y_train, y_test = train_test_split(data_file.iloc[:, :len(data_file.columns.tolist()) - 1],
                                                        data_file.iloc[:, -1],test_size=split_percentage)
    return x_train, x_test, y_train, y_test


def data_prepare(save_file_path,log_path,sub_sample_time,train_test_split_percentage):
    data = data_clean(TRAIN_PATH)
    # if if unbanlanced data you need do sample
    if (sub_sample_time != 0):
        data = sub_sample(data,sub_sample_time)
    if(train_test_split_percentage !=0):
        x_train, x_test, y_train, y_test = get_train_test_split(data,train_test_split_percentage)
    else:
        test = data_clean(TEST_PATH)
        x_test = test.iloc[:, :len(test.columns.tolist()) - 1]
        y_test = test.iloc[:, -1]
        x_train = data.iloc[:, :len(data.columns.tolist()) - 1]
        y_train = data.iloc[:, -1]
    if (save_file_path):
        csv_train = pd.concat([x_train, y_train], axis=1)
        csv_test = pd.concat([x_test, y_test], axis=1)
        csv_train.to_csv(save_file_path + 'train.csv', index=False)
        csv_test.to_csv(save_file_path + 'test.csv', index=False)

    fp = open(log_path+'log.txt','a')
    fp.write('trainData shape is '+str(x_train.shape)+'\n')
    fp.write('testData shape is '+str(x_test.shape)+'\n')
    fp.close()

    if(save_file_path):
        csv_train = pd.concat([x_train,y_train],axis = 1)
        csv_test = pd.concat([x_test,y_test],axis = 1)
        csv_train.to_csv(save_file_path + 'train.csv',index = False)
        csv_test.to_csv(save_file_path + 'test.csv',index = False)

    return  x_train, x_test, y_train, y_test

dir = '/home/yangqiao/pythonProject/Python_ML_Framework/log/'
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = data_prepare(None,dir,0,0)
    print x_train.shape
    print x_test.shape