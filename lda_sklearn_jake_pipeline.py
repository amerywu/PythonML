
from IPython import get_ipython

import logging

import dataloader
import datautils
import dataprocessing


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
############################################################
#              main program starts here
############################################################

# Set the important variables for data folder, data name, output folder
#dataDirectory="C:/Users/jake/Dropbox/Big Data Career/Data/tinysample"
dataDirectory="Z:/Dropbox/Big Data Career/Data/jd_bydocument/freq-2018-10-14_195000_by_45000"
#dataFile="fjd201808bysentence_merged"
dataFile="fjd201808byjob_removeHighAndLowFrequency_computer science"
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




logger.info("\n analyze \n\n")


ldaModel=dataprocessing.doLDA(data_tuple[0],
                              components=10,
                              maxiter=10,
                              learningmethod='online',
                              learningoffset=12,
                              randomstate=10,
                              verbose=1)
resultsTuple=dataprocessing.filterAndReportResultsLDA(ldaModel,columnMap,"ALL_TARGETS")
dataprocessing.saveListAsExcel(resultsTuple[0],outputDirectory,dataFile,resultsTuple[1])


######Important, use these lines of code below at the very end of any script that uses logging########

datautils.closeLogger

logging.shutdown()

SystemExit
print("Analysis Complete")
