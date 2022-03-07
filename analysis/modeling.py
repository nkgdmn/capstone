import nltk
import collections

from data_management.corpus import *
from data_management.document import *
from data_management.stopwords import *
from data_management.conditions import *

from analysis.analyses import *

from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import dataframe_image as dfi

import pandas as pd
import seaborn as sns
import networkx as nx
import numpy as np
import random
import math
import statistics

#Sillouette Scoring
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

#NMF (Non-negative Matrix Factorization) Modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

#LDA (Latent Dirichlet Allocation) Modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def corpus_to_list(corpus):
    '''
    corpus: corpus.Corpus -> List[string]
    Converts the corpus.Corpus object into a list of document texts. 
    Required conversion for topic modeling.
    
    '''
    list_of_texts = []
    for document in corpus: list_of_texts.append(document.text)
    return list_of_texts

def wc_dist(corpus):
    '''
    corpus: corpus.Corpus -> None
    Graphs the distribution of word counts of the documents in the
    given corpus. Recommended for use in large corpuses only.
    '''
    doc_lens = []
    for doc in corpus: doc_lens.append(doc.word_count)

    plt.figure(figsize=(10, 4), dpi=160)
    plt.hist(doc_lens, bins = 1000, color='navy')
    plt.text(800,  18, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(800,  16, "Median : " + str(round(np.median(doc_lens))))
    plt.text(800,  14, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(800,  12, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(800,  10, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=10)
    plt.xticks(np.linspace(0,1000,9))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=12))
    plt.tight_layout()
    plt.show()

def silhouette_score(corpus, n_features, stopword_list):
    '''
    corpus: List[string], n_features: int, stopword_list: List[string] -> None
    Finds the silhouette score for the given corpus.
    n_features: Length of TF-IDF vectors (NMF). Number of features (LDA). 
    '''
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words=stopword_list)
    #min_df: minimum number of documents that should contain feature (int)
    #max_df: maximum percentage a feature occurs within corpus (float)
    #words that occur in almost every document are  not suitable for classification 
    #b/c they do not provide any unique information about the document
    fit = vectorizer.fit_transform(corpus)
    #fit_transform converts text documents into corresponding numeric features
    kmeans = KMeans(random_state=1)
    visualizer = KElbowVisualizer(kmeans, k=(15, 25), metric='silhouette')
    visualizer.fit(fit)       
    visualizer.show()

def nmf(corpus, n_features, n_components, n_top_words, stopword_list, title):
    '''
    corpus: List[string], n_features: int, n_components: int, n_top_words: int, stopword_list: List[string], title: str) -> pd.DataFrame
    Returns dataframe with top ten words for each topic. 
    n_features: Length of TF-IDF vectors (NMF). Number of features (LDA). 
    n_components: Number of topics to model.
    n_top_words: Number of words to include for each topic, ranked by frequency.
    '''
    #below converts text into numbers (features)
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words=stopword_list)
    fit = vectorizer.fit_transform(corpus)
    nmf = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5).fit(fit)
    feature_names = vectorizer.get_feature_names()

    topics = {}
    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics[topic_idx+1] = top_features
    df = pd.DataFrame(topics)
    #numpy_array = df.to_numpy()
    #np.savetxt("test.txt", numpy_array, fmt = "%s", delimiter='\t')
    dfi.export(df, f'{title}_nmf.png')

    return topics

def lda(corpus: List[str], n_features: int, n_components: int, n_top_words: int, stopword_list: List[str], title: str) -> pd.DataFrame:
    '''
    corpus: List[string], n_features: int, n_components: int, n_top_words: int, stopword_list: List[string], title: str) -> pd.DataFrame
    Returns dataframe with top ten words for each topic. 
    n_features: Length of TF-IDF vectors (NMF). Number of features (LDA). 
    n_components: Number of topics to model.
    n_top_words: Number of words to include for each topic, ranked by frequency.
    '''
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words=stopword_list)    
    counts = vectorizer.fit_transform(corpus)    
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5, learning_method='online', learning_offset=50., random_state=0).fit(counts)
    feature_names = vectorizer.get_feature_names()
    
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics[topic_idx+1] = top_features
    df = pd.DataFrame(topics)
    dfi.export(df, f'{title}_lda.png')

    return topics

def graph_topic_categories(topic_categories, primary=None):
    '''
    topic_categories: List(tuple), primary: str -> None
    Takes a list of tuples in the form of (Primary Category, Secondary Category)
    for each topic in document or corpus. If primary category is specified
    (choose from 'C', 'A', 'P', 'S', 'L'), produces a graph of the 
    distribution of secondary categories to the primary category. If not,
    produces a graph of the distribution of primary categories.
    '''
    if primary:
        if primary not in ['C', 'A', 'P', 'S', 'L']: assert "Please input a valid category."
        secondary_categories = Counter([t[1] for t in topic_categories if t[0]==primary])
        for cat in ['C', 'A', 'P', 'S', 'L']:
            if cat not in secondary_categories.keys(): 
                secondary_categories[cat] = 0
        secondary_categories_sorted = sorted(secondary_categories.items())
    
        title = f'Distribution of Secondary Categories, Primary Category={primary}'
        categories = [t[0] for t in secondary_categories_sorted]
        counts = [t[1] for t in secondary_categories_sorted]

    else:
        primary_categories = Counter([t[0] for t in topic_categories])
        primary_categories_sorted = sorted(primary_categories.items())

        title = 'Distribution of Primary Categories'
        categories = [t[0] for t in primary_categories_sorted]
        counts = [t[1] for t in primary_categories_sorted]

    fig, ax = plt.subplots(figsize = (6,4))
        
    ax.set_title(title)
    ax.set_ylabel('Topic Count')
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(categories)

    if primary: ax.set_ylim([0, 30]) #arbitrary
    
    ax.bar(categories, counts)
    plt.show()  

def topic_by_document(corpus, n_features, n_components, stopword_list, model='nmf'):
    '''
    corpus: List(string), n_features: int, n_components: int, stopword_list: List[string], model: str -> None
    Finds which topic each document of the corpus most closely aligns with.
    Returns a Counter object with keys as topic numbers (0–(n_components-1))
    and values as counts representing number of documents.
    '''
    if model=='nmf':
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words=stopword_list)
        fit = vectorizer.fit_transform(corpus)
        nmf = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5).fit(fit)
        topics = nmf.transform(fit)
    elif model=='lda':
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words=stopword_list)    
        fit = vectorizer.fit_transform(corpus)    
        lda = LatentDirichletAllocation(n_components=n_components, max_iter=5, learning_method='online', learning_offset=50., random_state=0).fit(fit)
        topics = lda.transform(fit)
    else:
        assert "Please input a valid model type: 'nmf' or 'lda'."
    
    return Counter(topics.argmax(axis=1))

def graph_document_categories(topic_by_document, topic_by_category, corpus_name):
    '''
    topic_by_document: Counter(int, str), topic_by_category: Counter(int, int), corpus_name: str -> None
    Graphs the distribution of documents by category. 
    topic_by_document: Counter object with keys as topic numbers (0–(n_components-1))
    and values as counts representing number of documents.
    topic_by_category: Counter object with keys as topic numbers (0–(n_components-1))
    and values as categories (one of 'C', 'A', 'P', 'S', 'L').
    '''
    category_counts = Counter()
    category_counts['A']=0; category_counts['C']=0; category_counts['L']=0; category_counts['P']=0; category_counts['S']=0

    for topic in topic_by_document:
        count = topic_by_document[topic]
        category_counts[topic_by_category[topic]] += count
    
    title = f'Distribution of Categories, Corpus: {corpus_name}'
    categories = [c for c in category_counts]
    counts = [category_counts[c] for c in category_counts]

    fig, ax = plt.subplots(figsize = (6,4))
        
    ax.set_title(title)
    ax.set_ylabel('Document Count')
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(categories)

    ax.set_ylim([0, sum(topic_by_document.values())]) #arbitrary
    
    ax.bar(categories, counts)
    plt.show() 