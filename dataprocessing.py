

from matplotlib.pyplot import figure, show

from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd

from pandas import ExcelWriter

from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import coo_matrix

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


import dataloader
import datautils as du

#get_ipython().run_line_magic('matplotlib', 'inline')
import os

def elbow(X,outputDirectory,title,fromCount,toCount):
    from sklearn.cluster import KMeans
    from matplotlib import pyplot as plt
    distorsions = []
    for k in range(fromCount, toCount):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distorsions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(fromCount, toCount), distorsions)
    plt.grid(True)
    plt.title('Elbow curve')
    fig.savefig(outputDirectory+"/"+du.timeStamped()+title+"_figure.png")


#Converts data to TF_IDF
def convertToTfIdf(X):
     transformer = TfidfTransformer()
     transformedX=transformer.fit_transform(X)
     return(transformedX)



def doLDA(X,
          components=10,
          maxiter=500,
          learningmethod='online',
          learningoffset=0,
          randomstate=10,
          verbose=2):
     model = LatentDirichletAllocation(n_components=components,
                                       max_iter=maxiter,
                                       learning_method=learningmethod,
                                       learning_offset=learningoffset,
                                       random_state=randomstate,
                                       verbose=verbose).fit(X)

     return model

def filterAndReportResultsLDA(model,cmap,outputDirectory,target):


     listOfWordsByTopic = []
     n_top_words = 10
     for topic, comp in enumerate(model.components_):
         du.getLogger().debug("topic "+str(topic))
         du.getLogger().debug("comp " + str(comp))

         word_idx = np.argsort(comp)[::-1][:n_top_words]
         du.getLogger().debug(str(topic)+"word_idx" + str(word_idx))


         for i in word_idx:
             listOfWordsByTopic.append([topic, du.getTermByIdx(cmap,(i+1)), comp[i]])

     for i, (topic, term, value) in enumerate(listOfWordsByTopic):
         du.log().debug("topic "+str(topic)+" term "+str(term)+" value "+str(value))

     outputColumnNames=["topic","term","lda_weight"]

     return([listOfWordsByTopic,outputColumnNames])



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
    fig.savefig(outputDirectory+"/"+du.timeStamped()+title+"_figure.png")

def plotClusterCentroids(kmeans,outputDirectory,title="Cluster_Centroids"):
    #plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
    fig = figure()
    DefaultSize = fig.get_size_inches()
    DPI = fig.get_dpi()
    fig.set_size_inches( (DefaultSize[0]*5, DefaultSize[1]*5) )
    ax1 = fig.add_subplot(111)
    ax1.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
    fig.savefig(outputDirectory+"/"+du.timeStamped()+title+".png")




def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax



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

def saveListAsExcel(list,outputDirectory,fileName,outputColumnNames,targeName=""):
    outputDataframe = pd.DataFrame(list, columns=outputColumnNames)
    # Here we use a python function to convert our dataframe to Excel and then save it
    writer = ExcelWriter(outputDirectory + "/" + du.timeStamped() + targeName +"_"+fileName+ '.xlsx')
    outputDataframe.to_excel(writer, '_Sheet1')
    du.getLogger().debug("Saving to " + outputDirectory + "/" + du.timeStamped() + targeName +"_"+fileName+ '.xlsx')
    writer.save()

def subsetByCategory(X,y,targetIn):
    du.getLogger().debug("Original Set\n",X.__len__)
    n=0
    rows=list()
    for target in y:
        if target == targetIn:
            rows.append(n)
        n+=1
    array=np.asarray(rows)
    out=X[array,:]
    du.getLogger().debug("\n\nReturning SubSet\n" + str(out.__len__))
    dataloader.infoX(out)
    return(out)




def kmeansBySubset(X,y,columnMap,targetMap, clusterCount):
    for target in np.unique(y):
        targetName=du.getTargetByIdx(targetMap,target)
        du.getLogger().debug("\n\nSubset for "+str(targetName))
        subset = subsetByCategory(X,y,target)
        dataloader.infoX(subset)

        plot1(subset,targetName)
        kmeansForSubset=doKmeans(subset,clusterCount)
        filterAndReportResults(kmeansForSubset,columnMap,du.getTargetByIdx(targetMap,target),0.03)
        plotClusterCentroids(subset,kmeansForSubset,targetName)

#Converts data to TF_IDF
def convertToTfIdf(X):
     transformer = TfidfTransformer()
     transformedX=transformer.fit_transform(X)
     return(transformedX)

