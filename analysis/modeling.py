import nltk
import collections

from data_management.corpus import *
from data_management.document import *
from data_management.stopwords import *
from data_management.conditions import *
from analysis.gender_frequency import get_counts_by_pos, get_count_words
from analysis.instance_distance import words_instance_dist

from analysis.analyses import *

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

#NMF Modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

#LDA Modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#the process: text is converted into numbers (features)

#n_features: length of TF-IDF vectors to calculate, if using nmf model.
#generally, represents the number of words to include as features 

#n_components: number of topics to model
#n_top_words: number of words to print for each topic, ranked by frequency
#min_df: minimum number of documents that should contain feature (int)

#max_df: maximum percentage a feature occurs within corpus (float)
#(Words that occur in almost every document are  not suitable for classification b/c they do not provide any unique information about the document.)

#fit_transform: converts text documents into corresponding numeric features

#LDA==Latent Dirichlet Allocation
#NMF==Non-negative Matrix Factorization

def corpus_to_list(corpus):
    list_of_texts = []
    for document in corpus: list_of_texts.append(document.text)
    return list_of_texts

complete = Corpus('data/complete','data/complete/_metadata.csv')
rake = corpus_to_list(complete.subcorpus('performance_work', "rake's progress"))
pulcinella = corpus_to_list(complete.subcorpus('performance_work', 'pulcinella'))
danses = corpus_to_list(complete.subcorpus('performance_work', 'danses concertantes'))
firebird = corpus_to_list(complete.subcorpus('performance_work', 'firebird'))
apollo = corpus_to_list(complete.subcorpus('performance_work', 'apollo'))
concerto = corpus_to_list(complete.subcorpus('performance_work', 'concerto for piano and winds'))

def wc_dist(corpus):
    #layout only accomodates large corpuses
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
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words=stopword_list)
    fit = vectorizer.fit_transform(corpus)
    kmeans = KMeans(random_state=1)
    visualizer = KElbowVisualizer(kmeans, k=(15, 25), metric='silhouette')
    visualizer.fit(fit)       
    visualizer.show()

def nmf(corpus, n_features, n_components, n_top_words, stopword_list, title):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words=stopword_list)
    fit = vectorizer.fit_transform(corpus)
    nmf = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5).fit(fit)
    feature_names = vectorizer.get_feature_names()

    fig, axes = plt.subplots(2, 8, figsize=(10, 5), sharex=True)
    axes = axes.flatten()

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

def lda(corpus, n_features, n_components, n_top_words, stopword_list, title):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words=stopword_list)    
    counts = vectorizer.fit_transform(corpus)    
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5, learning_method='online', learning_offset=50., random_state=0).fit(counts)
    feature_names = vectorizer.get_feature_names()

    fig, axes = plt.subplots(2, 8, figsize=(10, 5), sharex=True)
    axes = axes.flatten()

    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics[topic_idx+1] = top_features
    df = pd.DataFrame(topics)
    dfi.export(df, f'{title}_lda.png')

    return topics

tc_rake = [('C', 'C'),('P', 'S'),('L', 'L'),('C', 'A'),('L', 'A'),('P', 'S'),('L', 'L'),('C', 'C'),('C', 'S'),('C', 'C'),('C', 'C'),('C', 'L'),('C', 'C'),('C', 'C'),('C', 'C')]
tc_pul = [('C', 'C'),('P', 'P'),('C', 'S'),('L', 'C'),('L', 'L'),('C', 'A'),('A', 'A'),('S', 'A'),('C', 'C'),('C', 'C'),('A', 'A'),('C', 'A'),('A', 'A'),('C', 'C'),('C', 'C')]
tc_danses = [('L', 'S'),('C', 'C'),('P', 'C'),('S','C'),('C', 'C'),('S', 'S'),('C', 'C'),('L', 'C'),('C', 'C'),('C', 'C'),('S', 'C'),('L', 'L'),('L', 'C'),('S', 'S'),('L', 'A')]
tc_fireb = [('L', 'C'),('C', 'L'),('C', 'P'),('C', 'L'),('S', 'C'),('C', 'C'),('C', 'S'),('C', 'C'),('S', 'L'),('C', 'S'),('P', 'S'),('C', 'P'),('C', 'C'),('S', 'C'),('L', 'L')]
tc_apollo = [('C', 'C'),('C', 'C'),('S', 'C'),('C', 'A'),('P', 'C'),('C', 'C'),('A', 'C'),('S', 'C'),('S', 'S'),('L', 'S'),('L', 'L'),('C', 'C'),('A', 'S'),('C', 'C'),('A', 'A')]
tc_concerto = [('A', 'A'),('S', 'C'),('C', 'C'),('A', 'P'),('L', 'C'),('C', 'C'),('C', 'C'),('A', 'C'),('P', 'S'),('L', 'C'),('C', 'A'),('A', 'A'),('C', 'A'),('A', 'A'),('S', 'C')]
topic_categories = tc_rake+tc_pul+tc_danses+tc_fireb+tc_apollo+tc_concerto

def graph_topic_categories(topic_categories, primary=None):
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

topic_by_category_rake = {0: 'C', 1: 'P', 2: 'L', 3: 'C', 4: 'L', 5: 'P', 6: 'L', 7: 'C', 8: 'C', 9: 'C', 10: 'C', 11: 'C', 12: 'C', 13: 'C', 14: 'C'}
topic_by_category_pul = {0: 'C', 1: 'P', 2: 'C', 3: 'L', 4: 'L', 5: 'C', 6: 'A', 7: 'S', 8: 'C', 9: 'C', 10: 'A', 11: 'C', 12: 'A', 13: 'C', 14: 'C'}
topic_by_category_dans = {0: 'L', 1: 'C', 2: 'P', 3: 'S', 4: 'C', 5: 'S', 6: 'C', 7: 'L', 8: 'C', 9: 'C', 10: 'S', 11: 'L', 12: 'L', 13: 'S', 14: 'L'}
topic_by_category_fire = {0: 'L', 1: 'C', 2: 'C', 3: 'C', 4: 'S', 5: 'C', 6: 'C', 7: 'C', 8: 'S', 9: 'C', 10: 'P', 11: 'C', 12: 'C', 13: 'S', 14: 'L'}
topic_by_category_apol = {0: 'C', 1: 'C', 2: 'S', 3: 'C', 4: 'P', 5: 'C', 6: 'A', 7: 'S', 8: 'S', 9: 'L', 10: 'L', 11: 'C', 12: 'A', 13: 'C', 14: 'A'}
topic_by_category_cpwi = {0: 'A', 1: 'S', 2: 'C', 3: 'A', 4: 'L', 5: 'C', 6: 'C', 7: 'A', 8: 'P', 9: 'L', 10: 'C', 11: 'A', 12: 'C', 13: 'A', 14: 'S'}

def topic_by_document(corpus, n_features, n_components, stopword_list, model='nmf'):
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

topic_by_document_rake = topic_by_document(rake, 1000, 15, stopwords_nltk+stopwords_general+stopwords_rake)
topic_by_document_pul = topic_by_document(rake, 1000, 15, stopwords_nltk+stopwords_general+stopwords_pul)
topic_by_document_dans = topic_by_document(rake, 1000, 15, stopwords_nltk+stopwords_general)
topic_by_document_fire = topic_by_document(rake, 1000, 15, stopwords_nltk+stopwords_general)
topic_by_document_apol = topic_by_document(rake, 1000, 15, stopwords_nltk+stopwords_general+stopwords_apollo)
topic_by_document_cpwi = topic_by_document(rake, 1000, 15, stopwords_nltk+stopwords_general+stopwords_cpwi)

def graph_document_categories(topic_by_document, topic_by_category, corpus_name):
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


