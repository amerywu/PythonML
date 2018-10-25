# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:05:34 2018

@author: jake
"""
from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


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


import os



def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

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



# main program starts here
        

# Set the important variables for data folder, data name, output folder

dataDirectory="Z:/Dropbox/Big Data Career/Data/jd_bydocument/freq-2018-10-14"
dataFile="fjd201808byjob_removeHighAndLowFrequency_education"
outputDirectory="C:/Users/jake/Desktop/"
timestamp=datautils.timeStamped()
datautils.createDirectory(outputDirectory)


# This creates a logger. You can use this to create an output file 
# which logs the steps of your analysis
logger=datautils.createLogger(outputDirectory)
logger.debug(getCurrentWorkingDirectory())
# load the data
data_tuple=dataloader.load(dataDirectory,dataFile)
columnMap=dataloader.loadColumnMap(dataDirectory,dataFile)
targetMap=dataloader.loadTargetMap(dataDirectory,dataFile)

########################################################################
#http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py

# Use tf (raw term count) features for LDA. data_tuple is already tf

#Apply LDA
#print("Fitting LDA models with tf features, "
#      "n_samples=%d and n_features=%d..."
#      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=20, max_iter=100,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0, verbose=1)
lda.fit(data_tuple[0])
top_n=12
for idx, topic in enumerate(lda.components_):
    logger.debug("Topic %d:" % (idx))
    print([(str(datautils.getTermByIdx(columnMap,[(i+1)])), str([i]), topic[i])
           for i in topic.argsort()[:-top_n - 1:-1]])






#########################################################################
#Neural network in topic modeling
#https://www.linkedin.com/pulse/neural-network-topic-modeling-manish-tripathi







