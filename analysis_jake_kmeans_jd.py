
from IPython import get_ipython

import logging

import dataloader
import datautils
import dataprocessing


import os
############################################################
#              main program starts here
#
############################################################

# Set the important variables for data folder, data name, output folder
#dataDirectory="C:/Users/jake/Dropbox/Big Data Career/Data/tinysample"
dataDirectory="Z:/Dropbox/Big Data Career/Data/tinysample"
#dataFile="fjd201808bysentence_merged"
dataFile="tiny"
outputDirectory="C:/users/jake/Desktop/outputFolder/"
timestamp=datautils.timeStamped()
datautils.createDirectory(outputDirectory)

##############################################
# This creates a logger. You can use this to create an output file
# which logs the steps of your analysis
##############################################
logger=datautils.createLogger(outputDirectory)


######################################################
# Now lets try with tfidf
######################################################

logger.info("\load data\n\n")

data_tuple=dataloader.load(dataDirectory,dataFile)

#The loaded column map
columnMap=dataloader.loadColumnMap(dataDirectory,dataFile)
#the loaded target map
targetMap=dataloader.loadTargetMap(dataDirectory,dataFile)

dataprocessing.elbow(data_tuple[0],outputDirectory,"elbow",2,8)
###############Run the kmeans and get back a model "kmeanstfidf"

kmeanstfidf=dataprocessing.doKmeans(
        data_tuple[0],
        clusterCount=2,
        maxIterations=2,
        init="k-means++",
        n_init=2,
        precompute_distances = 'auto',
        algorithm = 'auto',
        verbose=1)

logger.info("\nanalyze\n\n")
dataprocessing.filterAndReportResults(kmeanstfidf,columnMap,dataFile,outputDirectory,thresholdForColumnRemoval=0.3, thresholdForReporting=0.05)



######Important, use these lines of code below at the very end of any script that uses logging########

datautils.closeLogger
logging.shutdown()
SystemExit
