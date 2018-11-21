
from IPython import get_ipython

import logging

import dataloader
import datautils
import dataprocessing
import datamanipulation_conversions_jake as conversions
import kmeans_sklearn_jake as kmeansTools
import reporting_kmeansplots_sklearn_jake as kmeansplots
############################################################
#              main program starts here
############################################################

# Set the important variables for data folder, data name, output folder
#dataDirectory="C:/Users/jake/Dropbox/Big Data Career/Data/tinysample"
dataDirectory="Z:/Dropbox/Big Data Career/Data/jd_bydocument/freq-2018-11-01_300000_by_55000_95Percentile"
#dataFile="fjd201808bysentence_merged"
dataFile="fjd201810byjob_subset_life sciences95"
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

kmeansplots.elbow(data_tuple[0],outputDirectory,"elbow",2,8)
###############Run the kmeans and get back a model "kmeanstfidf"
X=conversions.convertToTfIdf(data_tuple[0])
kmeanstfidf=kmeansTools.doKmeans(
        X,
        clusterCount=10,
        maxIterations=10,
        init="k-means++",
        n_init=2,
        precompute_distances = 'auto',
        algorithm = 'auto',
        verbose=1)

logger.info("\n analyze \n\n")
reportTuple=kmeansTools.filterAndReportResults(kmeanstfidf,columnMap,dataFile, thresholdForReporting=0.05)
dataprocessing.saveListAsExcel(reportTuple[0],outputDirectory,dataFile,reportTuple[1])
logger.info("Making centroid plot")

kmeansplots.plotClusterCentroids(kmeanstfidf,outputDirectory)
logger.info("Done making centroid plot")
###### Important, use these lines of code below at the very end of any script that uses logging ########

datautils.closeLogger

logging.shutdown()


print("Analysis Complete")
SystemExit
