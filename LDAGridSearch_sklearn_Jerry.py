# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 23:15:57 2018

@author: Danjie
"""

%pwd
import os
os.chdir("F:/BigDataCareer/Git/PythonML")
%pwd

dataDirectory="F:/BigDataCareer/Data/jd_bydocument/freq-2018-10-16_300000_by_55000/"

dataFile="fjd201810byjob_removeHighAndLowFrequency_human resource management"

from sklearn.datasets import load_svmlight_file
X, y=load_svmlight_file(dataDirectory + dataFile + ".libsvm")
import pandas as pd
columnMap = pd.read_csv(dataDirectory + dataFile + "-columnMap.txt",header=None, names=("Idx","Term"))

targetMap = pd.read_csv(dataDirectory + dataFile + "-targetMap.txt",header=None, names=("Target","Idx"))

NUM_TOPICS=5

from sklearn.decomposition import LatentDirichletAllocation
# Build a Latent Dirichlet Allocation Model
lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=400, random_state=10,learning_method='batch', learning_decay=0, evaluate_every=1, perp_tol=0.01 , topic_word_prior=1/1000, verbose=1)

#lda_hr= lda_model.fit_transform(X)

lda_hr= lda_model.fit(X)
# checking model specification
lda_model.get_params()

def print_topics(model, feature_names, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(feature_names[i], round(topic[i],2))
        for i in topic.argsort()[:-top_n -1:-1]])
    
print("LDA Model:")
print_topics(lda_hr, columnMap['Term'])
print("=" * 40)


# grid search
from sklearn.model_selection import GridSearchCV
search_params = {'n_components': [2, 3, 4, 5, 6, 7]}
lda = LatentDirichletAllocation(n_components=2, topic_word_prior=1/1000, learning_method='batch', max_iter=500, evaluate_every=1, verbose=1, perp_tol= 0.01)

grid_search = GridSearchCV(estimator=lda, param_grid=search_params, verbose=1, cv=5)
grid_search.fit(X)


best_lda_model=grid_search.best_estimator_
print("Best Model's Params: ", grid_search.best_params_)

best_lda_model.get_params()

print("Best Log Likelihood Score: ", grid_search.best_score_)
print("Model Perplexity: ", best_lda_model.perplexity(X))


# show result
lda_output = best_lda_model.transform(X)
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
docnames = ["Doc" + str(i) for i in range(len(y))]

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
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

print("LDA Model:")    
display_topics(best_lda_model, columnMap['Term'],20)
print("=" * 40)

# showing topics with words and weights in tuples
# https://nlpforhackers.io/topic-modeling/

def print_topics(model, feature_names, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(feature_names[i], round(topic[i],2))
                        for i in topic.argsort()[:-top_n -1:-1]])
    
print("LDA Model:")
print_topics(best_lda_model, columnMap['Term'])
print("=" * 40)


# plotting

import matplotlib
import matplotlib.pyplot as plt
%matplotlib

n_topics = [2, 3, 4, 5, 6, 7]

# I'm not sure which ones are log-likelihoods
# please check more information in 
# grid_search.cv_results_
# if you're using an old version of SciKit Learn
# the result can be stored at
# grid_search.grid_scores_

log_likelihoods=[round(mean_score,2) for mean_score in grid_search.cv_results_['mean_train_score']]
plt.figure()
plt.plot(n_topics, log_likelihoods, label='0.7')
plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel("Log Likelihood Scores")
