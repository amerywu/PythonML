from sklearn.datasets import load_svmlight_file
import pandas as pd
import numpy as np  
import datautils

def greeting(name):
  print("Hello, " + name)
  
  
  
def infoX(X):
     logger=datautils.getLogger()
     logger.info("X.format"+ str(X.format))
     logger.info("X.dtype"+ str(X.dtype))
     logger.info("len(X.indices)"+ str(len(X.indices)))
     logger.info("X.ndim" + str(X.ndim))
     logger.info("X.__len__" + str(X.__len__))
     logger.info("X[:, 0].shape" + str(X[:, 0].shape))
    
def load(directory,dataFile):
     X, y = load_svmlight_file(directory+"/"+dataFile+'.libsvm')
     infoX(X)
     print("y.data.itemsize",y.data.itemsize)
     print("len(y)",len(y))
     print("y.ndim",y.ndim)
     print("np.unique(y)",np.unique(y))
     return (X,y)
 
def loadColumnMap(directory,dataFile):
    columnMap = pd.read_csv(directory+"/"+dataFile+'-columnMap.txt',header=None, names=("Idx","Term"))
    #print(columnMap)
    #print("\ncolumnMap\n")
    #print(columnMap.index)
    #print(columnMap.values)
    #print(columnMap.columns)
    #print(columnMap.dtypes)
    #print(columnMap.head)
    
    #print("\n\n")
    
    columnMap['Idx'] = columnMap['Idx'].str.replace('ColumnHeadTuple','')
    columnMap['Idx'] = columnMap['Idx'].str.replace('\(','')
    columnMap['Term'] = columnMap['Term'].str.replace('\)','')
    columnMap[['Idx']] = columnMap[['Idx']].apply(pd.to_numeric)
    columnMapSorted=columnMap.sort_values('Idx')
    return (columnMapSorted)

def loadTargetMap(directory,dataFile):
    targetMap = pd.read_csv(directory+"/"+dataFile+'-targetMap.txt',header=None, names=("Target","Idx"))


    targetMap['Idx'] = targetMap['Idx'].str.replace('\)','')
    targetMap['Target'] = targetMap['Target'].str.replace('\(','')
    targetMap[['Idx']] = targetMap[['Idx']].apply(pd.to_numeric)
    targetMapSorted=targetMap.sort_values('Idx')
    return(targetMapSorted)


def getCurrentWorkingDirectory():
    import os
    cwd = os.getcwd()
    return(str(cwd))