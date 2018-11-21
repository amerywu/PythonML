
from IPython import get_ipython

import logging
import lda_sklearn_jake as ldatools
import dataloader
import datautils
import dataprocessing

############################################################
#              main program starts here
############################################################

# Set the important variables for data folder, data name, output folder
#dataDirectory="C:/Users/jake/Dropbox/Big Data Career/Data/tinysample"
dataDirectory="Z:/Dropbox/Big Data Career/Data/jd_bydocument/freq-2018-11-01_300000_by_55000_99Percentile"
#dataFile="fjd201808bysentence_merged"
dataFile="fjd201810byjob_subset_finance99"
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
X=data_tuple[0]



logger.info("\n analyze \n\n")


ldaModel=ldatools.doLDA(X,
                              components=10,
                              maxiter=10,
                              learningmethod='online',
                              learningoffset=12,
                              randomstate=10,
                              verbose=1)
resultsTuple=ldatools.filterAndReportResultsLDA(ldaModel,columnMap)
dataprocessing.saveListAsExcel(resultsTuple[0],outputDirectory,dataFile,resultsTuple[1])

######Now repeat LDA analysis using grid search##########
n_topics = [2, 3, 4, 5, 6, 7]
gridReturnedTuple=ldatools.gridSearch(X,
               search_params={'n_components': n_topics},
               batch_size=128,
               doc_topic_prior=None,
               evaluate_every=5,
               learning_decay=0.0,
               learning_method='batch',
               learning_offset=10.0,
               max_doc_update_iter=100,
               max_iter=300,
               mean_change_tol=0.001,
               n_components=10,
               n_jobs=1,
               n_topics=None,
               perp_tol=0.1,
               random_state=None,
               topic_word_prior=1 / 1000,
               total_samples=1000000.0,
               verbose=1
               )


ldaModelFromGridSearch=gridReturnedTuple[0]
gridSearchModel=gridReturnedTuple[1]
lda_output = ldaModelFromGridSearch.transform(X)
topicnames = ["Topic" + str(i) for i in range(ldaModelFromGridSearch.n_components)]
docnames = ["Doc" + str(i) for i in range(len(data_tuple[1]))]
logger.info(topicnames)
logger.info(docnames)
ldatools.reportResults1(ldaModelFromGridSearch,X,data_tuple[1],columnMap,n_top_words = 10)

import matplotlib
import matplotlib.pyplot as plt




# I'm not sure which ones are log-likelihoods
# please check more information in
# grid_search.cv_results_
# if you're using an old version of SciKit Learn
# the result can be stored at
# grid_search.grid_scores_

log_likelihoods=[round(mean_score,2) for mean_score in gridSearchModel.cv_results_['mean_train_score']]
plt.figure()
plt.plot(n_topics, log_likelihoods, label='0.7')
plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel("Log Likelihood Scores")
plt.savefig(outputDirectory+"/"+timestamp+"Choosing Optimal LDA Model"+"_figure.png")
######Important, use these lines of code below at the very end of any script that uses logging########

datautils.closeLogger

logging.shutdown()

SystemExit
print("Analysis Complete")
