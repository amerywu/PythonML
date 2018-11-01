

from matplotlib.pyplot import figure, show

from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd

from pandas import ExcelWriter

from sklearn.feature_extraction.text import TfidfTransformer
import dataprocessing
import reporting_kmeansplots_sklearn_jake as kmeansplots

from sklearn.cluster import KMeans
import numpy as np



import dataloader
import datautils as du


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

#Converts data to TF_IDF
def convertToTfIdf(X):
     transformer = TfidfTransformer()
     transformedX=transformer.fit_transform(X)
     return(transformedX)

def predict(X, kmeans):
    labels = kmeans.predict(X)
    return labels

def kmeansBySubset(X,
                   y,
                   columnMap,
                   targetMap,
                   outputDirectory,
                   dataFile,
                   clusterCount=20,
                   maxIterations=300,
                   init="k-means++",
                   n_init="10",
                   precompute_distances='auto',
                   algorithm='auto',
                   verbose=1,
                   n_jobs=1,
                   thresholdForReporting=0.05):
    for target in np.unique(y):
        targetName=du.getTargetByIdx(targetMap,target)
        du.getLogger().debug("\n\nSubset for "+str(targetName))
        subset = dataprocessing.subsetByCategory(X,y,target)
        dataloader.infoX(subset)

        kmeansplots.sparsityPlot(subset,targetName)

        kmeansForSubset=doKmeans(subset,
                                 clusterCount,
                                 maxIterations,
                                 init,
                                 n_init,
                                 precompute_distances,
                                 algorithm,
                                 verbose,
                                 n_jobs)

        reportTuple = filterAndReportResults(subset, columnMap, dataFile, thresholdForReporting)
        dataprocessing.saveListAsExcel(reportTuple[0], outputDirectory, dataFile, reportTuple[1])

        kmeansplots.plotClusterCentroids(subset,kmeansForSubset,targetName)

def filterAndReportResults(kmeans,columnMap, target,thresholdForReporting=0.001):

    du.getLogger().debug(kmeans.cluster_centers_)
    outputColumnNames = ['Target','Cluster','Weight', 'Term']

    #This is the list of analysis results. We will add to it and then save as an Excel Spreadsheet
    listOfRows = []
    #columnsToRemove
    #This is a set of columns to remove. We will add to this list if our logic tells us to remove the column
    #NOTE a set cannot contain duplicates. This is good. We don't want to remove the same column twice!
    du.getLogger().debug(" K Means Parameter Values:")
    du.getLogger().debug(" inertia: "+str(kmeans.inertia_))
    du.getLogger().debug(" init: "+str(kmeans.init))
    du.getLogger().debug(" labels_: "+str(kmeans.labels_))
    du.getLogger().debug(" max_iter: "+str(kmeans.max_iter))
    du.getLogger().debug(" params: "+str(kmeans.get_params))
    du.getLogger().debug(" tol: "+str(kmeans.tol))
    du.getLogger().debug(" n_init: "+str(kmeans.n_init))

    clusterIdx = 0
    for row in kmeans.cluster_centers_:
        clusterIdx += 1
        du.getLogger().debug("\n------\nCluster\n\n")
        colIdx = 1
        for weight in row:
            if weight > thresholdForReporting:
                #This line simply prints the output to the console
                du.getLogger().debug(str(weight) + " col: "+ str(du.getTermByIdx(columnMap,colIdx)))
                # This adds the information about term and its weighting to a list.
                # We convert this list to a dataframe at the end of this function and save as excel
                # In other words we are going to save the results
                listOfRows.append([target,clusterIdx, weight, du.getTermByIdx(columnMap,colIdx)])



            colIdx += 1

    return ([listOfRows,outputColumnNames])