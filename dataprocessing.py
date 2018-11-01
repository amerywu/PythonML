

from matplotlib.pyplot import figure, show

from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd

from pandas import ExcelWriter

from sklearn.feature_extraction.text import TfidfTransformer

import reporting_kmeansplots_sklearn_jake as kmeansplots

from sklearn.cluster import KMeans
import numpy as np



import dataloader
import datautils as du










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







