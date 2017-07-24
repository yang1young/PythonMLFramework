
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#******************************************************************************************

def create():

    #create Series
    s = pd.Series([1,3,5,np.nan,6,8])
    print s

    #create dataframe
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
    print df

    #Creating a DataFrame by passing a dict of objects that can be converted to series-like.
    df2 = pd.DataFrame({ 'A' : 1.,
                        'B' : pd.Timestamp('20130102'),
                        'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                        'D' : np.array([3] * 4,dtype='int32'),
                        'E' : pd.Categorical(["test","train","test","train"]),
                        'F' : 'foo' })
    print df2
    #Having specific dtypes
    print df2.dtypes


#******************************************************************************************
def see():

    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
    print df

    #See the top & bottom rows of the frame'''
    print df.head(2)
    print df.tail(1)

    #Display the index, columns, and the underlying numpy data,num of line and col
    print df.index
    print df.columns
    print df.values
    print df.shape[0]
    print df.shape[1]

    #Describe shows a quick statistic summary of your data
    print df.describe()

    #Transposing your data
    print df.T

    #Sorting by an axis,0 is y,1 is x,ascending True is zhengxv,false is daoxv
    print df.sort_index(axis=0, ascending=False)

    #Sorting by values
    print df.sort(column='B')

    #see valuenums
    print df[0].value_counts()
    print df[u'hah'].value_counts()

    #see type and change
    df.dtypes
    df[['two', 'three']] = df[['two', 'three']].astype(float)


#******************************************************************************************

def selection():

    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
    print df

    #Selecting a single column, which yields a Series, equivalent to df.A
    print df['A']
    print df.A

    #Selecting via [], which slices the rows.
    print df[0:3]
    print df['20130102':'20130104']

    #Selection by Label

    #For getting a cross section using a label
    print df.loc[dates[0]]

    #Selecting on a multi-axis by label
    print df.loc[:,['A','B']]

    #Showing label slicing, both endpoints are included
    print df.loc['20130102':'20130104',['A','B']]

    #For getting a scalar value
    print df.loc[dates[0],'A']
    print df.at[dates[0],'A']


    #Selection by Position

    #Select via the position of the passed integers
    print df.iloc[3]

    #By integer slices, acting similar to numpy/python
    print df.iloc[3:5,0:2]

    #By lists of integer position locations, similar to the numpy/python style
    print df.iloc[[1,2,4],[0,2]]

    #For slicing rows explicitly
    print df.iloc[1:3,:]

    #For getting a value explicitly
    print df.iloc[1,1]
    print df.iat[1,1]


    #Boolean Indexing

    #Using a single column's values to select data.
    print df[df.A > 0]

    #Using the isin() method for filtering:
    df2 = df.copy()
    df2['E'] = ['one', 'one','two','three','four','three']
    print df2[df2['E'].isin(['two','four'])]

    #A where operation for getting.
    print df[df > 0]
    df2[df2 > 0] = -df2

    #Setting
    #Setting a new column automatically aligns the data by the indexes
    s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
    df['F'] = s1
    print df

    #Setting values by label/index
    df.at[dates[0],'A'] = 0
    df.iat[0,1] = 0
    print df

    #Setting by assigning with a numpy array
    df.loc[:,'D'] = np.array([5] * len(df))
    print df



#******************************************************************************************
# pandas primarily uses the value np.nan to represent missing data.
def missing():
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print df

    # Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of the data.
    df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
    df1.loc[dates[0]:dates[1], 'E'] = 1
    print df1

    # Filling missing data
    df2 = df1.fillna(value=5)
    print df2

    #To drop any rows that have missing data.
    df3 = df1.dropna(how='any')
    print df3

    #To get the boolean mask where values are nan
    print pd.isnull(df1)



#******************************************************************************************

def operate():
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print df

    #Performing a descriptive statistic
    print df.mean()
    print df.mean(1)

    #Applying functions to the data
    print df.apply(np.cumsum)
    print df.apply(lambda x: x.max() - x.min())

    #String Methods
    s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
    print s.str.lower()




#******************************************************************************************
def join():
    #concat():
    #Concatenating pandas objects together with concat():
    df = pd.DataFrame(np.random.randn(10, 4))
    print df

    pieces = [df[:3], df[3:7], df[7:]]
    pd.concat(pieces)

    #concact two dataFrame
    #data = pd.concat([data0,data1],ignore_index=True)

    #join
    left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
    right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
    print pd.merge(left, right, on='key')

    #Append
    df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
    print df
    s = df.iloc[3]
    print s
    df1 =  df.append(s, ignore_index=True)
    print df1

    #Grouping,group by
    #Splitting :the data into groups based on some criteria
    #Applying a function to each group independently
    #Combining the results into a data structure

    df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                           'foo', 'bar', 'foo', 'foo'],
                            'B' : ['one', 'one', 'two', 'three',
                            'two', 'two', 'one', 'three'],
                            'C' : np.random.randn(8),
                            'D' : np.random.randn(8)})
    print df

    print df.groupby('A').sum()

    print df.groupby(['A','B']).sum()

    #groupBy 2:
    i =0
    for data in df.groupby(df[3]):
        i=i+1
        print i
        print data

    #Time Series
    rng = pd.date_range('1/1/2012', periods=100, freq='S')
    print rng
    ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
    print ts
    print ts.resample('5Min').sum()



#******************************************************************************************

def plot():
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    print ts

    plt.figure()
    ts.plot()

    df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                      columns=['A', 'B', 'C', 'D'])
    df = df.cumsum()
    plt.figure(); df.plot(); plt.legend(loc='best')
    plt.show()


#******************************************************************************************

def file():
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                      columns=['A', 'B', 'C', 'D'])
    pd.read_csv('foo.csv')
    df.to_csv('foo.csv')

