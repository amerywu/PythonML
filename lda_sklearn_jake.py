from matplotlib.pyplot import figure, show

from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd

from pandas import ExcelWriter
from sklearn.feature_extraction.text import TfidfTransformer
import dataprocessing
import reporting_kmeansplots_sklearn_jake as kmeansplots
from sklearn.cluster import KMeans

import datautils as du
import numpy as np




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

def filterAndReportResultsLDA(model,cmap):


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




