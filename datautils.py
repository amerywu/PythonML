# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 13:03:39 2018

@author: jake
"""
import logging

import pandas as pd
import numpy as np  
from sklearn.cluster import KMeans  
from IPython import get_ipython
from sklearn.datasets import load_svmlight_file
from pandas import ExcelWriter
import os
import datetime
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer


def createLogger(directory):
    logger = logging.getLogger('ubc_merm_logger')
    if logger.handlers == []:
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        logfile = directory + "/logfile.txt"
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
    return(logger)
    
def getLogger():
    
    l=logging.getLogger('ubc_merm_logger')
    if l.handlers == []:
        l.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        logfile = "logfile.txt"
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        l.addHandler(fh)
        l.addHandler(ch)
    
    return (l)


def closeLogger():
    log=getLogger
    handlers = log.handlers[:]
    for handler in handlers:
        handler.close()
        log.removeHandler(handler)
        del handler
    logging.shutdown
    del log



def createDirectory(directory):
    if(not str(directory).endswith("/")):
        directory=directory+"/"

    if not os.path.exists(directory):
        os.makedirs(directory)    

def timeStamped():
    ts=str(datetime.datetime.now()).replace(" ","-").replace(".","-").replace(":","_")
    return (ts)


def getTermByIdx(cmap,n):
    onerow=cmap[cmap.Idx == n]
    onerow.iloc[0]['Term']
    return(onerow.iloc[0]['Term'])
    
def getTargetByIdx(tmap,n):
    onerow=tmap[tmap.Idx == n]
    onerow.iloc[0]['Target']
    return(onerow.iloc[0]['Target'])   
    
def convertToPandasDataFrame(X, y, colMap):
    sparseDataFrame = pd.SparseDataFrame(X)
    
    oldcolnames=list(sparseDataFrame.columns.values)
    newnames=list()
    for colname in oldcolnames:
        newnames.append(getTermByIdx(colMap,(colname+1)))
   
    sparseDataFrame.columns=newnames
    sparseDataFrame['aaatarget'] = y

    
    return(sparseDataFrame)




## Private method for internal use. No need to worry about this
def __dropColumnsFromPandasDataFrame(pddf,colsToRemove):
    pddf.drop(colsToRemove, inplace=True, axis=1)
    reducedColMap=__regenerateColumnMap(pddf)
    return (pddf,reducedColMap)

## Private method for internal use. No need to worry about this
def __regenerateColumnMap(pddf):
    count = 1
    newcols={}
    for name in pddf.columns.values:
        newcols[str(count)]=(count,name)

        count += 1
    newColMap=pd.DataFrame.from_dict(newcols, orient='index', columns=['Idx','Term'])
    
    return(newColMap)
    
  ## Private method for internal use. No need to worry about this  
def __convertFromPandasDataFrame(pdf):
    
    y= pdf['aaatarget'].tolist()
    del pdf['aaatarget']
    X=sparse.csr_matrix(pdf.to_coo())
    print(datetime.datetime.now())
    pdf['aaatarget'] = y
    return (X, y)

# See the extended comment on analysis_jake_kmeans.py. This method removes columns from a data matrix and 
# uodates the list of targets (y) and also the columnMap
def removeColumns(X,y,colmap, columnsToRemoveList):
    logger=getLogger()
    logger.debug(datetime.datetime.now())
    logger.debug("converting to dataframe")    
    pandasdf = convertToPandasDataFrame(X,y,colmap)
    logger.debug(datetime.datetime.now())
    logger.debug("removing")    
    pandasdfAndColMap=__dropColumnsFromPandasDataFrame(pandasdf,columnsToRemoveList)
    outdf=pandasdfAndColMap[0]
    outcolmap=pandasdfAndColMap[1]
    logger.debug(datetime.datetime.now())
    logger.debug("converting back to csr")
    convertedBackToCsr=__convertFromPandasDataFrame(outdf)
    newX=convertedBackToCsr[0]
    newy=convertedBackToCsr[1]
    return(newX, newy, outcolmap)
    


