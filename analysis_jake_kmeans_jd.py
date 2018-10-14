# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:05:34 2018

@author: jake
"""
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np  
from sklearn.cluster import KMeans  
from IPython import get_ipython
from sklearn.datasets import load_svmlight_file
from pandas import ExcelWriter
import datetime
from sklearn.feature_extraction.text import TfidfTransformer
import logging

import dataloader
import datautils

get_ipython().run_line_magic('matplotlib', 'inline')
import os



def plot1(X, title="figure"):
    fig = figure()
    DefaultSize = fig.get_size_inches()
    DPI = fig.get_dpi()
    fig.set_size_inches( (DefaultSize[0]*5, DefaultSize[1]*5) )
    ax1 = fig.add_subplot(111)
    ax1.spy(X, precision=1, markersize=0.1, marker=".",aspect="auto")
    fig.savefig(outputDirectory+"/"+timestamp+title+"_figure.png")

def plot2(X,kmeans,title="cluster"):
    #plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
    fig = figure()
    DefaultSize = fig.get_size_inches()
    DPI = fig.get_dpi()
    fig.set_size_inches( (DefaultSize[0]*5, DefaultSize[1]*5) )
    ax1 = fig.add_subplot(111)
    ax1.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
    fig.savefig(outputDirectory+"/"+timestamp+title+".png")
    
    
def doKmeans(X,clusterCount=10):
    kmeans = KMeans(n_clusters=clusterCount, init='random', max_iter=100, n_init=1, verbose=1)
    kmeans.fit(X)  
    #logger.debug("Cluster count ", kmeans.n_clusters)
    #logger.debug("Cluster centers ",kmeans.cluster_centers_)  
    #logger.debug(kmeans.labels_)  
    #ccount = kmeans.n_clusters
    #ccenters = kmeans.cluster_centers_
    return(kmeans)
    
 

def predict(X, kmeans):
    labels = kmeans.predict(X)
    return labels

# This is (nonsense) logic that helps you decide whether to remove a column. You will need to rewrite this code
# so that it performs whatever logic operation you require.
# IMPORTANT. It does NOT remove the column. It simply answers true or false as to whether it should be removed
def removeColumnDecision( weight,  threshold=0, otherVariable1=0,otherVariable2=0):
    if weight > threshold:
        if otherVariable1 * otherVariable2 >= 0:
            return (True)
        else:
            return (False)
    else:
        return(False)
            
    
    
    
    

def filterAndReportResults(kmeans,columnMap, target, thresholdForColumnRemoval, threshold):
    
    logger.debug(kmeans.cluster_centers_)
    outputColumnNames = ['Target','Cluster','Weight', 'Term']
    
    #This is the list of analysis results. We will add to it and then save as an Excel Spreadsheet
    listOfRows = []
    #columnsToRemove
    #This is a set of columns to remove. We will add to this list if our logic tells us to remove the column
    #NOTE a set cannot contain duplicates. This is good. We don't want to remove the same column twice!
    columnsToRemove = set()
    
    clusterIdx = 0
    for row in kmeans.cluster_centers_:
        clusterIdx += 1
        logger.debug("\n------\nCluster\n\n")
        colIdx = 1
        for weight in row:
            if weight > threshold:
                #This line simply prints the output to the console
                logger.debug(str(weight) + " col: "+ str(datautils.getTermByIdx(columnMap,colIdx)))
                # This adds the information about term and its weighting to a list.
                # We convert this list to a dataframe at the end of this function and save as excel
                # In other words we are going to save the results
                listOfRows.append([target,clusterIdx, weight, datautils.getTermByIdx(columnMap,colIdx)])
                #Now we decide whether we want to remove the column.
                shouldWeRemoveTheColumnFromTheDataSet=removeColumnDecision(weight,thresholdForColumnRemoval)
                #If remove == True, then we will add the name of the column to our list of columns to remove
                if(shouldWeRemoveTheColumnFromTheDataSet == True):
                    columnsToRemove.add(datautils.getTermByIdx(columnMap,colIdx))
                
            colIdx += 1
     
    #Here we convert our list of analyzed results to a dataframe
    outputDataframe = pd.DataFrame(listOfRows, columns=outputColumnNames)
    # Here we use a python function to convert our dataframe to Excel and then save it
    writer = ExcelWriter(outputDirectory+"/"+datautils.timeStamped()+target+'_kmeansOutput.xlsx')
    outputDataframe.to_excel(writer,'kmeansOutput_Sheet1')
    logger.debug("Saving to "+outputDirectory+"/"+datautils.timeStamped()+'kmeansOutput.xlsx')
    writer.save()
    #Finally we return our list of columns that we want to remove from the dataset. You can use this in
    # another function to actualy remove the columns.
    return (columnsToRemove)

def subsetByCategory(X,y,targetIn):
    logger.debug("Original Set\n",X.__len__)
    n=0
    rows=list()
    for target in y:
        if target == targetIn:
            rows.append(n)
        n+=1    
    array=np.asarray(rows)
    out=X[array,:]
    logger.debug("\n\nReturning SubSet\n" + str(out.__len__))
    dataloader.infoX(out)
    return(out)
    
            
def getCurrentWorkingDirectory():
    import os
    cwd = os.getcwd()
    return(str(cwd))
        
def kmeansBySubset(X,y,columnMap,targetMap, clusterCount):
    for target in np.unique(y):
        targetName=datautils.getTargetByIdx(targetMap,target)
        logger.debug("\n\nSubset for "+str(targetName))
        subset = subsetByCategory(X,y,target)
        dataloader.infoX(subset)
        
        plot1(subset,targetName)
        kmeansForSubset=doKmeans(subset,clusterCount)
        filterAndReportResults(kmeansForSubset,columnMap,datautils.getTargetByIdx(targetMap,target),0.03)
        plot2(subset,kmeansForSubset,targetName)

#Converts data to TF_IDF
def convertToTfIdf(X):
     transformer = TfidfTransformer()
     transformedX=transformer.fit_transform(X)
     return(transformedX)

# FOR REFERENCE ONLY
# This block of code demonstrates how to remove columns
# It's just a test! Use function removeColumns in real life.
    
# Remember! We need to remove columns in the csr data matrix.
# This means that all the column numbers change
# This means we must change the Idx column in columnMap too.
# Also, what happens if you remove ALL the columns (i.e., words) in
# a document. Then rows will disappear. If a row disappears then y
# (our list of targets) must also be changed.
# One mistake in any of this and the entire data set is corrupted.

# The best way to do this is to use a pandas data frame ( Imagine
# this is an excel spreadsheet in computer memory).
# We add all the X data to the dataframe. We also create a new
# column in the dataframe for all of the targets (y). Then we 
# change the columnheads so that they are the actual words/terms
# rather than the column numbers. The we will have ALL our data
# in one place. 
# The function convertToPandasDataFrame(X,y,colmap) performs the
# above task. 
# NOTE: The underlying structure of the dataframe is sparse so it will
# not use too much memory.
# Then we remove columns from this dataframe. We do this by creating a list
# of column NAMES (not numbers)
# We use the function __dropColumnsFromPandasDataFrame. We give the
# function two arguments. (1) the datframe we just created and 
# (2) the list of column names we will drop.
# __dropColumnsFromPandasDataFrame returns a new dataframe AND
# a new columnMap. Be very careful to only use the new dataframe and columnMap
# or you will get very confusing results.
# We still have a problem! kmeans wants a csr matrix, not a datframe
# so now we take our dataframe with columns removed and we
# call the function __convertFromPandasDataFrame(outdf).
# This returns a csr matrix with X the frequency data and 
# y the list of targets.
    
def testColumnRemoval(X,y,colmap):
    pandasdf = datautils.convertToPandasDataFrame(X,y,colmap)
    logger.debug(datetime.datetime.now())
    logger.debug("removing")
    colNames=list(pandasdf.columns.values)
    colsToRemove=set()
    colsToRemove.append(colNames[1])
    colsToRemove.append(colNames[4])
    colsToRemove.append(colNames[7])
    
    pandasdfAndColMap=datautils.__dropColumnsFromPandasDataFrame(pandasdf,colsToRemove)
    outdf=pandasdfAndColMap[0]
    outcolmap=pandasdfAndColMap[1]
    logger.debug(datetime.datetime.now())
    logger.debug("converting back to csr")
    convertedBackToCsr=datautils.__convertFromPandasDataFrame(outdf)
    newX=convertedBackToCsr[0]
    newy=convertedBackToCsr[1]
    return(newX, newy, outcolmap)

# main program starts here
        

# Set the important variables for data folder, data name, output folder
#dataDirectory="C:/Users/jake/Dropbox/Big Data Career/Data/tinysample"  
dataDirectory="C:/Users/jake/Dropbox/Big Data Career/Data/jd_bysentence"      
dataFile="fjd201808bysentence_merged"
#dataFile ="fjd201808bysentence_trimmed"
outputDirectory="C:/Users/jake/Desktop/Output"
timestamp=datautils.timeStamped()
datautils.createDirectory(outputDirectory)

# This creates a logger. You can use this to create an output file 
# which logs the steps of your analysis
logger=datautils.createLogger(outputDirectory)
logger.debug("My first Python log message.")
# load the data
data_tuple=dataloader.load(dataDirectory,dataFile)
columnMap=dataloader.loadColumnMap(dataDirectory,dataFile)
targetMap=dataloader.loadTargetMap(dataDirectory,dataFile)


# plot the data
plot1(data_tuple[0])
logger.debug("Plot saved to "+ outputDirectory)


logger.info("\nkmeans 1\n\n")

# run kmeans
kmeans=doKmeans(data_tuple[0],13)

logger.info("\nanalyze1\n\n")
# Analyze the kmeans results. This will also give us a list of columns we want to remove from the dataset
setOfColumnsToRemoveAfterAnalysis=filterAndReportResults(kmeans,columnMap,"ALL",0.2, 0.03)
logger.debug("setOfColumnsToRemoveAfterAnalysis\n"+str(setOfColumnsToRemoveAfterAnalysis))
#plot2(data_tuple[0],kmeans)
#
# Now let's get a new dataset. Our previous analysis told us to remove some columns. 
# We will get a new dataset with columns removed
logger.info("\nremove columns\n\n")
dataWithColumnsRemovedTupleOfThree=datautils.removeColumns(data_tuple[0], data_tuple[1],columnMap,setOfColumnsToRemoveAfterAnalysis)

#The above lines gave us 3 NEW items.  A new datamatrix X. A new list of targets y.  A new columnMap,
dataWithColumnsRemoved_X=dataWithColumnsRemovedTupleOfThree[0]
dataWithColumnsRemoved_y=dataWithColumnsRemovedTupleOfThree[1]
dataWithColumnsRemoved_columnMap=dataWithColumnsRemovedTupleOfThree[2]
dataloader.infoX(dataWithColumnsRemoved_X)
# So now we can repeat our analysis with the new dataset. Run kmeans on data with columns removed
logger.info("\nkmeans 2\n\n")
kmeans1=doKmeans(dataWithColumnsRemoved_X,10)
# Get the results from data with columns removed and maybve get more columns to remove
moreColumnsToRemove=filterAndReportResults(kmeans1,dataWithColumnsRemoved_columnMap,"ALL",0.2, 0.02)


# run kmeans by subset
#kmeansBySubset(data_tuple[0],data_tuple[1],columnMap,targetMap,10)



######################################################
# Now lets try with tfidf
######################################################

logger.info("\nrelaod\n\n")

data_tupleFresh=dataloader.load(dataDirectory,dataFile,1234)
columnMapFresh=dataloader.loadColumnMap(dataDirectory,dataFile)
targetMapFresh=dataloader.loadTargetMap(dataDirectory,dataFile)
logger.info("\ntfidf\n\n")
tfidfX=convertToTfIdf(data_tupleFresh[0])
kmeanstfidf=doKmeans(tfidfX,12)
logger.info("\nanalyze\n\n")
setOfColumnsToRemoveAfterAnalysisWithTFIDF=filterAndReportResults(kmeanstfidf,columnMapFresh,"ALLwithtfidf",0.3, 0.002)
logger.debug("setOfColumnsToRemoveAfterAnalysisWithTFIDF "+str(setOfColumnsToRemoveAfterAnalysisWithTFIDF))



dataWithColumnsRemovedAfterTFIDFTupleOfThree=datautils.removeColumns(tfidfX, data_tupleFresh[1],columnMapFresh,setOfColumnsToRemoveAfterAnalysisWithTFIDF)
dataWithColumnsRemovedAfterTFIDF_X=dataWithColumnsRemovedAfterTFIDFTupleOfThree[0]
dataWithColumnsRemovedAfterTFIDF_y=dataWithColumnsRemovedAfterTFIDFTupleOfThree[1]
dataWithColumnsRemovedAfterTFIDF_columnMap=dataWithColumnsRemovedAfterTFIDFTupleOfThree[2]
dataloader.infoX(dataWithColumnsRemovedAfterTFIDF_X)


######Important, use these lines of code below at the very end of any script that uses logging########

datautils.closeLogger
logging.shutdown()
SystemExit

















     
     
     





 


