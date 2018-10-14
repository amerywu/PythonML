

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

#get_ipython().run_line_magic('matplotlib', 'inline')
import os




#Converts data to TF_IDF
def convertToTfIdf(X):
     transformer = TfidfTransformer()
     transformedX=transformer.fit_transform(X)
     return(transformedX)


#######DO NOT CHANGE THIS CODE##################
def doKmeans(X,
             clusterCount=20,
             maxIterations=300,
             init="k-means++",
             n_init="10",
             precompute_distances = 'auto',
             algorithm = 'auto',
             verbose=1,
             n_jobs= 1):
    kmeans = KMeans(n_clusters=clusterCount,
                    init='k-means++',
                    n_jobs= n_jobs,
                    max_iter=maxIterations,
                    n_init=n_init,
                    verbose=verbose,
                    precompute_distances = precompute_distances,
                    algorithm = algorithm)
    kmeans.fit(X)

    return(kmeans)



def plot1(X, outputDirectory,title="figure"):
    fig = figure()
    DefaultSize = fig.get_size_inches()
    DPI = fig.get_dpi()
    fig.set_size_inches( (DefaultSize[0]*5, DefaultSize[1]*5) )
    ax1 = fig.add_subplot(111)
    ax1.spy(X, precision=1, markersize=0.1, marker=".",aspect="auto")
    fig.savefig(outputDirectory+"/"+datautils.timeStamped()+title+"_figure.png")

def plot2(X,kmeans,outputDirectory,title="cluster"):
    #plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
    fig = figure()
    DefaultSize = fig.get_size_inches()
    DPI = fig.get_dpi()
    fig.set_size_inches( (DefaultSize[0]*5, DefaultSize[1]*5) )
    ax1 = fig.add_subplot(111)
    ax1.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
    fig.savefig(outputDirectory+"/"+datautils.timeStamped()+title+".png")






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






def filterAndReportResults(kmeans,columnMap, target, outputDirectory,thresholdForColumnRemoval=0.3, thresholdForReporting=0.001):

    logger.debug(kmeans.cluster_centers_)
    outputColumnNames = ['Target','Cluster','Weight', 'Term']

    #This is the list of analysis results. We will add to it and then save as an Excel Spreadsheet
    listOfRows = []
    #columnsToRemove
    #This is a set of columns to remove. We will add to this list if our logic tells us to remove the column
    #NOTE a set cannot contain duplicates. This is good. We don't want to remove the same column twice!
    columnsToRemove = set()
    logger.debug(" inertia: "+str(kmeans.inertia_))
    logger.debug(" init: "+str(kmeans.init))
    logger.debug(" labels_: "+str(kmeans.labels_))
    logger.debug(" max_iter: "+str(kmeans.max_iter))
    logger.debug(" params: "+str(kmeans.get_params))
    logger.debug(" tol: "+str(kmeans.tol))
    logger.debug(" n_init: "+str(kmeans.n_init))

    clusterIdx = 0
    for row in kmeans.cluster_centers_:
        clusterIdx += 1
        logger.debug("\n------\nCluster\n\n")
        colIdx = 1
        for weight in row:
            if weight > thresholdForReporting:
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

logger=datautils.getLogger()