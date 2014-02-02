from __future__ import division
import nltk
import random
import re, pprint, os
import numpy
import nltk.corpus
from nltk.corpus import brown
from nltk import cluster
from nltk.cluster import util
from nltk.cluster import api
from nltk.cluster import euclidean_distance
from nltk.cluster import cosine_distance

texts = brown
print "Read in", len(texts.fileids()), "documents..."
numdoc=len(texts.fileids())
print "The first five are:", texts.fileids()[:5]
doc_id=0
unique_terms = list(set(texts.words()))
idf_count=[]
for word in unique_terms:
    idf_count.append(0)
print "Found a total of", len(unique_terms), "unique terms"



# Function to create a BOW for one document.  This is called with the fileid
# of one of the files in the corpus.  We convert its list of words into an
# nltk.Text object so we can use the count method.  Then for each of
# our unique words, we have a feature which is the count for that word
def BOW(document,_id):
    #print type(document)
    document = nltk.Text(texts.words((document)))
    word_counts = []
    for word in unique_terms:#tf calc
        word_counts.append(document.count(word)/len(document))
        if(document.count(word)>0):
            idf_count[unique_terms.index(word)]+=1
    #_id+=1
   # print "done",_id
    return word_counts

#function to add in the idf metric to the vectors, not 100% if it works right
def IDF(vectors):
    idfvectors=[]
    #cant iterate over int need to use something else
    for v in vectors:
        for n in idf_count:#index out of range
            idfvectors[vectors.index(v)][idf_count.index(n)]=vectors[vectors.index(v)][idf_count.index(n)]*numpy.log(numdoc/n)
    return idfvectors

# And here we actually call the function and create our list of tf vectors.
vectors = [numpy.array(BOW(f,doc_id)) for f in texts.fileids()]
print "TFVectors created."
#final_vectors=IDF(vectors)
print "IDFVectors created."
print "First 10 words are", unique_terms[:10]
print "First 10 counts for first document are", vectors[0][0:10]
# We now have a vector ready to feed to our clusterer of choice.
#more rigourus mean selection possibly needed 
means= [vectors[200],vectors[350],vectors[170],vectors[300],vectors[25],vectors[100],vectors[250],vectors[450],vectors[400],vectors[40]]
#setup the initial means
#currently testing with no inital means, uncomment below to change this
clusterer = cluster.KMeansClusterer(10, cosine_distance)#,initial_means=means)

clusters = clusterer.cluster(vectors, True, trace=True)
#go cluster with kmeans
print'clustering done'
print 'Clustered:', vectors
print 'As:', clusters
print 'Means:', clusterer.means()
#clusters.plot()


