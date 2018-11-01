
import pandas as pd
import datautils
import gensim
import scipy
from sklearn.feature_extraction.text import TfidfTransformer

def convertToPandasDataFrame(X, y, colMap):
    sparseDataFrame = pd.SparseDataFrame(X)

    oldcolnames=list(sparseDataFrame.columns.values)
    newnames=list()
    for colname in oldcolnames:
        newnames.append(datautils.getTermByIdx(colMap,(colname+1)))

    sparseDataFrame.columns=newnames
    sparseDataFrame['aaatarget'] = y


    return(sparseDataFrame)



def convertToGensimCorporaAndDictionary(X,columnMap):
    dct = gensim.corpora.Dictionary()

    datautils.getLogger().info("\n convertToGensimCorporaAndDictionary \n\n")

    cx = scipy.sparse.coo_matrix(X)

    corpora = []
    doc = []
    currentRow = 0
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if (i > currentRow):
            print("-> ")
            corpora.append(doc)
            doc = []
            currentRow = i
        # print("(%d, %d), %s" % (i,j,v))
        for x in range(0, int(v)):
            doc.append(datautils.getTermByIdx(columnMap, j + 1))

    dct.add_documents(corpora)
    common_corpus = [dct.doc2bow(text) for text in corpora]
    return(common_corpus,dct)

#Converts data to TF_IDF
def convertToTfIdf(X):
     transformer = TfidfTransformer()
     transformedX=transformer.fit_transform(X)
     return(transformedX)