
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import NMF, LatentDirichletAllocation


import datautils as du
import numpy as np


#Courtesy of Jerry
def gridSearch(X,
               search_params={'n_components': [2, 4, 6, 8, 10]},
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
               ):
    lda = LatentDirichletAllocation()

    # major parameter setting
    # evaluate_every=5
    # learning_method='batch'
    # learning_decay=0.0
    # According to SK learn documentation, when the value is 0.0 and batch_size is n_samples, the update method is same as batch learning.
    # scoring='accuracy', verbose=5

    # it seems what I specified here doesn't work.
    # when the iteration finished, it will print out GridSearchCV() specification, which is different from what I've specified.
    # This is an issue to be solved.

    GridSearchCV(cv=5, error_score='raise',
                 estimator=LatentDirichletAllocation(

                                                     batch_size=batch_size,
                                                     doc_topic_prior=doc_topic_prior,
                                                     evaluate_every=evaluate_every,
                                                     learning_decay=learning_decay,
                                                     learning_method=learning_method,
                                                     learning_offset=learning_offset,
                                                     max_doc_update_iter=max_doc_update_iter,
                                                     max_iter=max_iter,
                                                     mean_change_tol=mean_change_tol,
                                                     n_components=n_components,
                                                     n_jobs=n_jobs,
                                                     n_topics=n_topics,
                                                     perp_tol=perp_tol,
                                                     random_state=random_state,
                                                     topic_word_prior=topic_word_prior,
                                                     total_samples=total_samples,
                                                     verbose=verbose),
                 fit_params=None,
                 iid=True,
                 n_jobs=1,
                 param_grid=search_params,
                 pre_dispatch='2*n_jobs',
                 refit=True,
                 return_train_score='warn',
                 scoring='accuracy',
                 verbose=5)

    # setting learning_decay=0

    gridSearchModel = GridSearchCV(lda, param_grid=search_params)
    gridSearchModel.fit(X)

    # best result from grid search
    best_lda_model = gridSearchModel.best_estimator_
    return ([best_lda_model, gridSearchModel])

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

def filterAndReportResultsLDA(model,cmap,n_top_words=10):


     listOfWordsByTopic = []

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



def reportResults1(best_lda_model,X,y,columnMap,n_top_words = 10):
    lda_output = best_lda_model.transform(X)
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
    docnames = ["Doc" + str(i) for i in range(len(y))]
    du.getLogger().debug(topicnames)
    du.getLogger().debug(docnames)
    import numpy as np
    import pandas as pd
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    # showing document*topic table with a column called dominant_topic
    df_document_topic.head(15)

    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")

    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    df_topic_distribution

    df_topic_keywords = pd.DataFrame(best_lda_model.components_)
    df_topic_keywords.columns = columnMap['Term']
    df_topic_keywords.index = topicnames
    df_topic_keywords.head()

    # showing topics with words, but witout weights
    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            du.getLogger().debug("Topic %d:" % (topic_idx))
            du.getLogger().debug(" ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))

    du.getLogger().debug("LDA Model:")
    display_topics(best_lda_model, columnMap['Term'], 20)
    du.getLogger().debug("=" * 40)

    # showing topics with words and weights in tuples
    # https://nlpforhackers.io/topic-modeling/



    du.getLogger().debug("LDA Model:")
    print_topics(best_lda_model, columnMap['Term'],n_top_words)
    du.getLogger().debug("=" * 40)


def print_topics(model, feature_names, n_top_words=10):
    for idx, topic in enumerate(model.components_):
        du.getLogger().debug("Topic %d:" % (idx))
        du.getLogger().debug([(feature_names[i], round(topic[i], 2))
            for i in topic.argsort()[:-n_top_words - 1:-1]])

