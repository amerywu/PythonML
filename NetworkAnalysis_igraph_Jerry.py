# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 18:38:10 2018

@author: Danjie
"""
# setting data directory
dataDirectory="F:/BigDataCareer/Data/jd_bydocument/freq-2018-10-16_300000_by_55000/"

# setting data file: marketing
dataFile="fjd201810byjob_removeHighAndLowFrequency_marketing"

# setting data file: Human Resource
dataFile="fjd201810byjob_removeHighAndLowFrequency_human resource management"

# loading original marketing data sets
from sklearn.datasets import load_svmlight_file
X, y=load_svmlight_file(dataDirectory + dataFile + ".libsvm")
import pandas as pd
columnMap = pd.read_csv(dataDirectory + dataFile + "-columnMap.txt",header=None, names=("Idx","Term"))

targetMap = pd.read_csv(dataDirectory + dataFile + "-targetMap.txt",header=None, names=("Target","Idx"))

# converting original data set to network data format, i.e., nodes, edges, and weights
# taking marketing data file as example
# step1: converting original data set to pandas data frame by using Jake's functions in datautils.py 

import pandas as pd
from datautils import getTermByIdx
from datautils import convertToPandasDataFrame
mkt_df=convertToPandasDataFrame(X,y,columnMap)
# step2: dropping last column, which is the code for "marketing"
mkt_df.drop(columns='aaatarget',inplace=True)

# step3: calculating edge weight
# if a pair of words appear in the same document, then the weight count adds 1
# in each document, the occurance of every pair of words is recorded, then saved in a list
# after all documents are processed, counting the frequency of every pair of words in the list, which is the weight of edge

import itertools
import collections
# input a dataframe, get a list of weightd edges
# each edge is stored as a 3-element tuple
def GetWeightedEdge(df):
    edge_list=[]
    for i in range(0,len(df.index)):
        row=df.iloc[i,:]
        reduced_row=row[row.isnull()==False]
        Combinations=list(itertools.combinations(reduced_row.index,2))
        edge_list.extend(Combinations) 
    edge_frequency= collections.Counter(edge_list)
    keys=list(edge_frequency.keys())
    values=list(edge_frequency.values())
    return list(map(lambda key,val:key+(val,),keys, values))

# converting pandas data frame to a list of edges with weight
# this step will take a while
mkt_edge_w= GetWeightedEdge(mkt_df)

# step4: setting up a graph (i.e. network) by using igraph
import igraph as ig
# creating an empty graph
mkt=ig.Graph()
# adding nodes (vertices) to the graph
# nodes coming from data frame column names, in our case, they are words in marketing data set
mkt.add_vertices(mkt_df.columns.values)
# adding edges to the graph
# edge only, without weight, because edges and weights are added separately in igraph
mkt_edge= [x[:-1] for x in mkt_edge_w]
mkt.add_edges(mkt_edge)
# adding edge weight as an attribute of edge
mkt_weight= [x[-1:] for x in mkt_edge_w]
mkt_weight= [x[0] for x in mkt_weight]
mkt.es['weight'] = mkt_weight
# adding edge label as another attribute, in case we need show edge names, otherwise, the graph will only show numbers to represent edges 
mkt.es['label']= mkt_edge
# checking graph summary
mkt.summary()
# checking graph density as a whole
mkt.density() 

# have a look at several elements of node list, edge list and weight list
# v denotes vertex
mkt.vs['name'][:5]
mkt.get_edgelist()[0:10]
# e denotes edge
mkt.es['weight'][:5]
mkt.es['label'][:10]

# community detection
# you can check community detection methods by typing "mkt.community" in IPython console, then press "Tab" key on your keyboard, it will show all the available methods
# here, only multilevel algorithm is taken as an example
# this method is pretty fast
mkt_com_ml= mkt.community_multilevel(weights='weight')
# checking how many communities we've got
mkt_com_ml.sizes()
# getting every community as a subgraph from the original marketing graph
mkt_com_ml_sub00= mkt_com_ml.subgraph(0)
mkt_com_ml_sub01= mkt_com_ml.subgraph(1)
mkt_com_ml_sub02= mkt_com_ml.subgraph(2)
mkt_com_ml_sub03= mkt_com_ml.subgraph(3)
mkt_com_ml_sub04= mkt_com_ml.subgraph(4)

# alternative
# getting all communities and save it to a list
mkt_coms= mkt_com_ml.subgraphs()

# this function get top nodes with highest degree for all communities in a graph, and return a list of node lists
# if the number specified is larger than total number of nodes in a community, then the total number is used instead
def TopNodes(community_list, number):
    ls=[]
    for idx,item in enumerate(community_list):
        dictionary= dict(zip(item.vs['name'], ig.Graph.strength(item, weights='weight')))
        sorted_list=sorted(dictionary.items(), key=lambda kv: kv[1], reverse=True)
        if number>item.vcount():
            import warnings
            warnings.warn("Number %d is larger than total number of nodes in community %d, total number of %d is used instead." % (number, idx, item.vcount()))
            ls.append(sorted_list)
        else:
            ls.append(sorted_list[:number])
    return(ls)

top20_nodes= TopNodes(mkt_coms, 20)
top20_nodes[0]
top20_nodes[4]

        
# you can also check it one by one
# checking nodes with highest and lowest degree
# ig.Graph.strength function computes the degree of nodes
# the result is a dictionary
sub00_dict= dict(zip(mkt_com_ml_sub00.vs['name'],ig.Graph.strength(mkt_com_ml_sub00, weights='weight')))
# sorting nodes by degree by descending order
# the result is a list of tuples
sub00_dict_sorted= sorted(sub00_dict.items(), key=lambda kv: kv[1], reverse=True)

# top 20 nodes with highest degree 
sub00_dict_sorted[:20]
# top 20 nodes with lowest degree 
sub00_dict_sorted[-20:]


# this function get top edges with highest weight for all communities in a graph, and return a list of edge lists.
# if the number specified is larger than total number of edges in a community, then the total number is used instead.
def TopEdges(community_list, number):
    ls=[]
    for idx,item in enumerate(community_list):
        dictionary= dict(zip(item.es['label'], item.es['weight']))
        sorted_list=sorted(dictionary.items(), key=lambda kv: kv[1], reverse=True)
        if number>item.ecount():
            import warnings
            warnings.warn("Number %d is larger than total number of nodes in community %d, total number of %d is used instead." % (number, idx, item.ecount()))
            ls.append(sorted_list)
        else:
            ls.append(sorted_list[:number])
    return(ls)

top20_edges= TopEdges(mkt_coms, 20)
top20_edges[0]

# you can also check it one by one
# checking edges with highest and lowest weights, which is similar to top nodes
sub00_e_dict= dict(zip(mkt_com_ml_sub00.es['label'],mkt_com_ml_sub00.es['weight']))

sub00_e_dict_sorted= sorted(sub00_e_dict.items(), key=lambda kv: kv[1], reverse=True)

sub00_e_dict_sorted[:20]
sub00_e_dict_sorted[-20:]




# plotting top nodes and edges
# note: cairo and pycairo should be installed in advance before you can use igraph.plot function
# "Graph plotting in igraph is implemented using a third-party package called Cairo. If you want to create publication-quality plots in igraph on Windows, you must also install Cairo and its Python bindings. "
# http://igraph.org/python/doc/tutorial/install.html
# "The Cairo project does not provide pre-compiled binaries for Windows", which means even you've download the Cairo package, for example, cairo-1.16.0.tar.xz, you cannot install it in your Windows system. so I found the easiest way to install Cairo is installing GTK+ platform, which incorporates Cairo already.
# the detailed instruction for installing GTK+ :
# https://www.gtk.org/download/windows.php

# if you use other system rather than Windows, you can download Cairo and install it following the official instruction
# official Cairo download and installation instruction
# https://www.cairographics.org/download/


# you can download pycairo from the following web page, which is a 'whl' format file
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycairo
# then use pip install command to install in a prompt, for example, "pip install pycairo-1.18.0-cp36-cp36m-win_amd64.whl" 

# after installing Cairo and pycairo, you can try "import cairo" and "import pycairo" in IPython sonsole, if it is successful without any warnings, then you can go ahead to use igraph.plot function. 
# note: you don't have to import cairo and pycairo in order to use igraph.plot(), just install them properly

# this is a simple function to plot top nodes (V) or top edges (E) in a batch
def PlottingTopVorE(OriginalCommunities,List,path=''):
    import igraph as ig
    for idx,item in enumerate(List):
        VorE= [i[0] for i in item]
        if isinstance(VorE[0], str): #checking the element in the list is a node
            sub= OriginalCommunities[idx].subgraph(VorE)
            ig.plot(sub, layout='circular',vertex_label=sub.vs['name'],vertex_size=2,vertex_label_size=30, edge_label=None,bbox=(1500, 1500), margin=50, target=path+'top_nodes_com_'+str(idx)+'.png')
        else: # the element in the list is an edge
            sub= OriginalCommunities[idx].subgraph_edges(VorE)
            ig.plot(sub, layout='circular',vertex_label=sub.vs['name'],vertex_size=2,vertex_label_size=30, edge_label=None,bbox=(1500, 1500), margin=50, target=path+'top_edges_com_'+str(idx)+'.png')
    return()

# This function uses earlier results from above codes
# mkt_coms is a list of communities (graphs)
# top20_nodes is a list of lists returned from TopNodes()
# top20_edges is a list of lists returned from TopEdges()
# you can also specify the directory where the plots would be saved
    

PlottingTopVorE(mkt_coms, top20_nodes)
PlottingTopVorE(mkt_coms, top20_edges, path='F:/BigDataCareer/Python/')

# note: the plot may not properly show the name of nodes in IPython when a single plot is produced(at least, in Spyder), so I save the plot as a png picture in the working directory, then the names of nodes are properly displayed  

# community 0
# top 20 nodes with highest degree
# extracting top 20 nodes from the list
top20_v_00=[i[0] for i in sub00_dict_sorted[:20]]
# getting a subgraph of 20 nodes
sub00_top20_v=mkt_com_ml_sub00.subgraph(top20_v_00)
# plotting
ig.plot(sub00_top20_v, layout='circular', vertex_label=sub00_top20_v.vs['name'], vertex_size=2,vertex_label_size=30, edge_label=None,bbox=(1500, 1500), margin=50, target='sub00_top20_v.png')

sub00_top20_v.density()
# top 20 edges with highest weight
# extracting top 20 edges from the list
top20_e_00=[i[0] for i in sub00_e_dict_sorted[:20]]
# getting a subgraph of 20 edges
sub00_top20_e=mkt_com_ml_sub00.subgraph_edges(top20_e_00)
# plotting
ig.plot(sub00_top20_e, layout='circular', vertex_label=sub00_top20_e.vs['name'], vertex_size=2,vertex_label_size=30, edge_label=None,bbox=(1500, 1500), margin=50, target='sub00_top20_e.png')

