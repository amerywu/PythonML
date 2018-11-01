
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


def sparsityPlot(X, outputDirectory,title="figure"):
    fig = figure()
    DefaultSize = fig.get_size_inches()
    DPI = fig.get_dpi()
    fig.set_size_inches( (DefaultSize[0]*5, DefaultSize[1]*5) )
    ax1 = fig.add_subplot(111)
    ax1.spy(X, precision=1, markersize=0.1, marker=".",aspect="auto")
    fig.savefig(outputDirectory+"/"+du.timeStamped()+title+"_figure.png")


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