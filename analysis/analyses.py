import nltk
import collections

from data_management.corpus import *
from data_management.document import *
from data_management.stopwords import *
from data_management.conditions import *
from analysis.gender_frequency import get_counts_by_pos, get_count_words
from analysis.instance_distance import words_instance_dist

from nltk.util import ngrams
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import PunktSentenceTokenizer

#graphs n stuff
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.colors as colors

import seaborn as sns
import networkx as nx
import numpy as np
import random
import math
import statistics

sia = SentimentIntensityAnalyzer()
stemmer = PorterStemmer()
wnl = WordNetLemmatizer()
pst = PunktSentenceTokenizer()
  
# helper functions

def remove_duplicates(corpus):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object

    Returns
    -------
    the cleaned corpus 

    >>> len(complete)
    448
    >>> len(remove_duplicates(complete))
    412
    '''

    base_names = []
    for document in corpus:
        base = document.filename.split('.')[0] 
        if base not in base_names: base_names.append(base)
        else: corpus.documents.remove(document) #remove as this is a duplicate

    return corpus

def _get_wordnet_tag(nltk_tag):
    '''
    Parameters
    ----------
    nltk_tag : the nltk part-of-speech tag
    
    Returns
    -------
    the wordnet tag (str)
    '''

    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else: #default categorize by adjective tag        
        return wordnet.ADJ

def get_sentences(corpus, filename):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    filename (str) : the name of the txt file 

    Returns
    -------
    dictionary with keys as sentence number and values as sentences

    >>> get_sentences(complete, 'bradley, lionel4.txt')
    {1: "Stravinsky's Apollon Musagete...", 2: 'years ago at the Bath Assembly.', 
    3: "I don't think it...", 4: 'This is a work which has a calm...'}
    '''

    try:
        document = corpus.get_document('filename', filename) #get document
    except ValueError: #document not in corpus
        raise

    clean = Document._clean_quotes(document.text) #get rid of weird quotation marks
    clean_again = pst.tokenize(clean) #tokenized sentences will be free of extraneous quotes
    
    sentences = {}
    for i in range(len(clean_again)):
        sentences[i+1] = clean_again[i]
    return sentences

def get_identities(word):
    '''
    Parameters
    ----------
    word (str) : the word whose equivalents are to be found

    Returns
    -------
    list of word equivalents (including input word)

    >>> get_identities('stravinsky')
    ['stravinsky', 'stravinski', 'strawinsky', 'strawinski']
    '''

    try:
        return [word]+identities[word]
    except KeyError:
        #print('No identities found. Check hierarchy or input new entry.')
        return [word]

def get_roots(corpus):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object

    Returns
    -------
    a dictionary of roots as keys and list of associated words as values
    
    >>> get_roots(complete)
    {'musaget': ['musagetes', 'musagete'], 
    'wonder': ['wonder', 'wondering', 'wondered'], 
    'form': ['form', 'forms', 'formed', 'formative'],...}
    '''

    words = list(corpus.get_wordcount_counter())
    roots = {}

    for word in words:
            #stem the word
            stem = stemmer.stem(word)

            if stem != word: 
                #stemming is successful; that is, stem is shorter or different than the word itself
                roots[stem] = roots.get(stem, list())
                roots[stem].append(word)
            else: 
                #stemming is unsuccessful; lemmatize the word
                nltk_tag = nltk.pos_tag([word])[0][1]
                lem = wnl.lemmatize(word, _get_wordnet_tag(nltk_tag))
                
                roots[lem] = roots.get(lem, list())
                roots[lem].append(word)

    return roots

def get_ngrams(corpus, filename, n, word=None, stopword_lists=None):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    filename (str) : filename associated with a document in the corpus
    n (int) : number of grams
    word (str) : word to search within the document, optional
    stopword_lists (list) : list of custom stopword lists, optional

    Returns
    -------
    a list of tuples of length n

    >>> get_ngrams(complete, 'lafond, martine.txt', 5, 'stravinsky', [stopwords_nltk])
    [('also', 'magnificently', 'performed', 'dissonance', 'stravinsky'), 
    ('stravinsky', 'could', 'well', 'eliminated', 'special'), 
    ('culture', 'sitting', 'bitter', 'end', 'stravinsky'), 
    ('stravinsky', 'rite', 'spring', 'last', 'saturday')]
    '''

    #normalize input
    n = int(n) 
    if word == 'None': word = None 
    if stopword_lists == 'None': stopword_lists = None

    try:
        document = corpus.get_document('filename', filename) #get document
    except ValueError: #document not in corpus
        raise 

    tokens = document.get_tokenized_text()

    #step i. sanity check on the format of stopword_lists
    if stopword_lists and len(stopword_lists)!=0: #empty list means no stopword list provided
        if (type(stopword_lists)==str): #for command line
            try:
                stopword_lists = eval(stopword_lists)
            except NameError: #not a list
                'Argument takes a list of stopword lists. Check input and try again.'
        assert(type(stopword_lists[0])==list), 'Argument takes a list of stopword lists. Check input and try again.'

        #step ii. if there is only one list, that is the master list
        if len(stopword_lists)==1:
            stopwords_master = set(stopword_lists[0])

        else: #step iii. multiple lists; create master stopword list
            stopwords_master = set()
            for l in stopword_lists: 
                stopwords_master = stopwords_master.union(set(l))
    
        tokens_clean = [t for t in tokens if t not in stopwords_master]

        #print(f'{len(tokens)-len(tokens_clean)} words were omitted.')

    else: tokens_clean = tokens

    result = list(ngrams(tokens_clean, n))

    if word != None and len(word)!=0: 
        result_clean = [] #only keep tuples where the first or last element is the word
        for t in result: 
            if word==t[0] or word==t[-1]: result_clean.append(t) #the other tuples are repeats; do this to avoid double-counting

        assert(len(result_clean)!=0), 'No instance of word in document.'

        return result_clean

    return result

# general functions

def get_words_by_root(corpus, word, check_identities=True):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    word (str) : the input word
    check_identities (bool) : find word equivalents first, default is True

    Returns
    -------
    the list of associated words; None if no words associated
    
    >>> get_words_by_root(complete, 'dissonance')
    ['dissonances', 'dissonance', 'dissonant']

    >>> get_words_by_root(complete, 'stravinsky')
    ['stravinsky', 'stravinskys', 'strawinsky']
    
    >>> get_words_by_root(complete, 'stravinsky', False)
    ['stravinsky', 'stravinskys']
    '''

    #normalize input
    if type(check_identities)==str: check_identities = eval(check_identities)

    if check_identities:
        word_equals = get_identities(word) 
    else: word_equals = [word]

    word_list = []
    roots = get_roots(corpus)

    #take advantage of the fact that roots and words must start w/ the same letter
    #sub_r = {key: r[key] for key in r if key[0] == word[0]}

    #assume the worst case; that is, each identity belongs to a different root
    for w in word_equals: 
        stem = stemmer.stem(w)
        try: 
            word_list+=roots[stem] #add words of the same root to list (including word itself)
        except KeyError:
            try: 
                nltk_tag = nltk.pos_tag([w])[0][1]
                lem = wnl.lemmatize(w, _get_wordnet_tag(nltk_tag))
                word_list+=roots[lem] #add words of the same root to list (including word itself)
            except KeyError:
                pass #no hits

    if len(word_list)==0:
        print('No such word, its equivalents, or associated words in corpus.')
        return

    return list(set(word_list)) #get rid of duplicates, should they occur

def get_wordcount_counter(corpus, stopword_lists=None, part_of_speech_tag=None):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    stopword_lists (list) : list of custom stopword lists, optional
    part_of_speech_tag (str) : part of speech tag (use base tags), optional
    
    Returns
    -------
    a list of tuples in the form of (word, count) in descending order

    >>> get_wordcount_counter(complete, [stopwords_nltk], 'JJ')
    Counter({'musical': 338, 'new': 266, 'much': 205, 'last': 186, 'great': 170, ...})
    '''
    
    #normalize input
    if part_of_speech_tag == 'None': part_of_speech_tag = None
    if stopword_lists == 'None': stopword_lists = None

    word_counter = corpus.get_wordcount_counter()

    #step i. sanity check on the format of stopword_lists
    if stopword_lists and len(stopword_lists)!=0: #empty list means no stopword list provided
        if (type(stopword_lists)==str): #for command line
            try:
                stopword_lists = eval(stopword_lists)
            except NameError: #not a list
                'Argument takes a list of stopword lists. Check input and try again.'
        assert(type(stopword_lists[0])==list), 'Argument takes a list of stopword lists. Check input and try again.'

        #step ii. if there is only one list, that is the master list
        if len(stopword_lists)==1:
            stopwords_master = set(stopword_lists[0])

        else: #step iii. multiple lists; create master stopword list
            stopwords_master = set()
            for l in stopword_lists: 
                stopwords_master = stopwords_master.union(set(l))

        word_counter_clean = collections.Counter()
        for word in word_counter:
            if word not in stopwords_master: 
                word_counter_clean[word] = word_counter[word]

        #print(f'{len(word_counter)-len(word_counter_clean)} words were omitted.')

        word_counter = word_counter_clean

    for i, d in dependencies: #i=independent; d=dependent
        c_i = word_counter[i]
        c_d = word_counter[d]
        #print(f'original count of independent term "{i}" is {c_i}')
        #print(f'original count of dependent term "{d}" is {c_d}')
        if c_d != 0:
            if c_d <= c_i:
                word_counter[i] -= c_d #subtract counts in i by d
                word_counter[d] = 0 #same thing as saying word_counter[d] -= c_d
                #print(f'new count of independent term "{i}" is {word_counter[i]} (minus {c_d})')
            else: pass #no action taken
                #print(f'count of dependent word "{d}" exceeds count of independent term "{i}" (no action).')
        
    if not part_of_speech_tag:
        return word_counter

    else:
        pos_counter_all = get_counts_by_pos(word_counter)
        
        pos_counter = collections.Counter()
        for pos in pos_counter_all:
            if pos[:2] == part_of_speech_tag: #get subtags
                pos_counter += pos_counter_all[pos]

        return pos_counter

def get_wordcount_by_field(corpus, metadata_field, word=None, stopword_lists=None, part_of_speech_tag=None, check_identities=True): 
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    metadata_field (str) : the metadata_field to consider
    word (str) : the word (and its associates) to count, optional
    stopword_lists (list) : list of custom stopword lists, optional
    part_of_speech_tag (str) : part of speech tag (use base tags), optional
    check_identities (bool) : find word equivalents first, default is True
    
    Returns
    -------
    a dictionary with the metadata field entries as keys and counts as values 
    
    >>> get_wordcount_by_field(complete, 'performance_work', 'dissonance')
    {'dissonances': {'concerto for two pianos': 1}, 
    'dissonant': {'circus polka': 1, 'danses concertantes': 2, 'mass': 1, 'persephone': 1, 
                'pulcinella': 1, 'rite of spring': 2, 'scherzo a la russe': 1}, 
    'dissonance': {'concerto for piano and winds': 1, 'general': 5, 'mass': 1, 
                'pulcinella': 1, 'rite of spring': 3, 'symphony of psalms': 2}}
    '''

    #normalize input
    if word == 'None': word = None

    by_word = {}

    values = corpus.get_field_vals(metadata_field)  

    #omit the documents whose metadata entry is blank
    if '' in values: values.remove('')
    
    for value in values:
        #create subcorpuses for each entry of metadata field
        current_subcorpus = corpus.subcorpus(metadata_field, value)  
        current_subcorpus.name = value

        words_and_counts = get_wordcount_counter(current_subcorpus, stopword_lists, part_of_speech_tag)
        for w in words_and_counts:
            by_word[w] = by_word.get(w, {})
            by_word[w][current_subcorpus.name] = words_and_counts[w]

    if word: 
        associates = get_words_by_root(corpus, word, check_identities)
        assert(associates), 'No such word or associated words in corpus.'
        print(f'Associated words: {associates}')
        
        subset = {}
        for a in associates:
            try: subset[a] = by_word[a]
            except KeyError: pass
        return subset

    return by_word

# distance analyses

def get_word_windows(corpus, word, window=5, stopword_lists=None, part_of_speech_tag=None, check_identities=True):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    word (str) : the word to form window around
    window (int) : number of words around the input to consider, default is 5
    stopword_lists (list) : list of custom stopword lists, optional
    part_of_speech_tag (str) : part of speech tag (use base tags), optional
    check_identities (bool) : find word equivalents first, default is True

    Returns
    -------
    a dictionary of filenames as keys and list of neighboring words as values
    
    >>> get_word_windows(complete, 'dissonance', 3, [stopwords_nltk])
    {'armitage, merle3.txt': ['preoccupation', 'empty', 'intellectualizing', 'nonsensical'], 
    'berger, arthur11.txt': ['allusion', 'polarity'], 
    'berger, arthur16.txt': ['shuns', 'argue'], 
    'downes, olin5.txt': ['broadly', 'grinding', 'fine', 'speaking'],...}

    >>> get_word_windows(complete, 'dissonance', 3, [stopwords_nltk])['goldberg, albert.txt']
    ['dry', 'logical', 'later']
    '''

    #normalize input
    window = int(window) 
    if stopword_lists == 'None': stopword_lists = None
    if part_of_speech_tag == 'None': part_of_speech_tag = None
    if type(check_identities)==str: check_identities = eval(check_identities)

    omitted = []
    grand_total = {}

    #step i. sanity check on the format of stopword_lists
    if stopword_lists and len(stopword_lists)!=0: #empty list means no stopword list provided
        if (type(stopword_lists)==str): #for command line
            try:
                stopword_lists = eval(stopword_lists)
                assert(type(stopword_lists[0])==list), 'Argument takes a list of stopword lists. Check input and try again.'
            except NameError: 'Argument takes a list of stopword lists. Check input and try again.'
        assert(type(stopword_lists[0])==list), 'Argument takes a list of stopword lists. Check input and try again.'

        #step ii. if there is only one list, that is the master list
        if len(stopword_lists)==1:
            stopwords_master = set(stopword_lists[0])

        else: #step iii. multiple lists; create master stopword list
            stopwords_master = set()
            for l in stopword_lists: 
                stopwords_master = stopwords_master.union(set(l))
 
    for document in corpus:

        neighbor_counter = collections.Counter()

        #step 1. find word locations in document 
        tokens = document.get_tokenized_text()
        indices = []

        if check_identities:
            word_equals = get_identities(word) #accounting for word equivalence
        else: word_equals = [word]

        hits = sum(list(get_count_words(document, word_equals).values()))
        if hits == 0: 
            omitted.append((document.filename, "No match (all)."))
            continue #move onto next document

        i = 0
        while (i < len(tokens)) and (len(indices) < hits):
            if tokens[i] in word_equals: 
                indices.append(i)
                #print((document.filename, tokens[i]))
            i += 1

        #step 2. find adjacent terms
        for index in indices:
            for i in range(window):
                if index-i-1 < 0: break #index is 0
                else: neighbor_counter.update([tokens[index-i-1]]) #since i starts at 0  
            
            for i in range(window):
                try: neighbor_counter.update([tokens[index+i+1]]) #since i starts at 0
                except IndexError: break #last index

        #step 3. determine if pos tag present
        #step 4. remove stopwords
        if not part_of_speech_tag:
            if stopword_lists:
                set_neighbors = set(neighbor_counter) #set of all neighboring words, no counts
                cleaned = set_neighbors.difference(stopwords_all)
                grand_total[document.filename] = list(cleaned)
            else: grand_total[document.filename] = list(neighbor_counter)

        else:
            neighbor_by_pos = collections.Counter()
            #get dictionary of parts of speech as keys and counter objects as values
            neighbor_by_pos_all = get_counts_by_pos(neighbor_counter)
        
            #collect all target pos into another counter
            for pos in neighbor_by_pos_all:   
                if pos[:2] == part_of_speech_tag: #get all subtags
                    neighbor_by_pos += neighbor_by_pos_all[pos] #add the counter object
            
            #only append counter objects to the grand_total if they aren't empty
            if len(neighbor_by_pos) != 0:
                if stopword_lists:
                    set_neighbors = set(neighbor_by_pos)
                    cleaned = set_neighbors.difference(stopwords_all)
                    grand_total[document.filename] = list(cleaned)
                else:
                    grand_total[document.filename] = list(neighbor_by_pos)
            else:
                omitted.append((document.filename, "No match (POS)."))
    
    #print('The following documents were omitted from this analysis.')
    #print(omitted)
    
    #step 5. ensure we are returning something of value
    assert(len(grand_total) > 0), f"There are no instances of words in the specified window of '{word}.'"

    #step 6. print out quantified version of result for ease of future analysis
    # print('***Below are the quantified results.***')
    # for filename in grand_total:
    #     print(f'{filename}: {len(grand_total[filename])}')
    # print()

    return grand_total

def get_ww_instance(corpus, word, window=5, stopword_lists=None, part_of_speech_tag=None, check_identities=True):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    word (str) : the word to form window around
    window (int) : number of words around the input to consider, default is 5
    stopword_lists (list) : list of custom stopword lists, optional
    part_of_speech_tag (str) : part of speech tag (use base tags), optional
    check_identities (bool) : find word equivalents first, default is True

    Returns
    -------
    a dictionary with neighboring words as keys and lists as values containing 
    filenames in which the words appear
    
    >>> get_ww_instance(complete, 'dissonance', 3, [stopwords_nltk])
    {'nonsensical': ['armitage, merle3.txt'], 
    'intellectualizing': ['armitage, merle3.txt'], 
    'preoccupation': ['armitage, merle3.txt'], 
    'empty': ['armitage, merle3.txt'], 
    'allusion': ['berger, arthur11.txt'], 
    'polarity': ['berger, arthur11.txt'],...}

    >>> get_ww_instance(complete, 'dissonance')['stravinsky']
    ['berger, arthur16.txt', 'goldberg, albert.txt', 'lafond, martine.txt']
    '''

    by_filename = get_word_windows(corpus, word, window, stopword_lists, part_of_speech_tag, check_identities)
    by_neighbor = {}
    
    for filename in by_filename:
        doc = corpus.get_document('filename', filename)

        neighbors = by_filename[filename]
        for neighbor in neighbors: 
            by_neighbor[neighbor] = by_neighbor.get(neighbor, list())
            by_neighbor[neighbor].append(filename)

    return by_neighbor 

def get_ww_by_root(corpus, word, window=5, stopword_lists=None, part_of_speech_tag=None, check_identities=True):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    word (str) : the word to form window around
    window (int) : number of words around the input to consider, default is 5
    stopword_lists (list) : list of custom stopword lists, optional
    part_of_speech_tag (str) : part of speech tag (use base tags), optional
    check_identities (bool) : find word equivalents first, default is True

    Returns
    -------
    a dictionary with associated words as keys and dictionaries as values 
    in the format of what is returned by get_ww_instance

    >>> get_ww_by_root(complete, 'dissonance', 3, [stopwords_nltk])
    {'dissonances': {
        'lines': ['berger, arthur17-2.txt'], 
        'abstraction': ['berger, arthur17-2.txt'], 
        'angular': ['berger, arthur17-2.txt']}, 
    'dissonant': {
        'artists': ['1944jones.txt'], 
        'angular': ['1944jones.txt'], 
        'elaborate': ['1944jones.txt'],...},
    'dissonance': {
        'empty': ['armitage, merle3.txt'], 
        'preoccupation': ['armitage, merle3.txt'], 
        'intellectualizing': ['armitage, merle3.txt'],...}}

    >>> get_ww_by_root(complete, 'dissonance')['dissonant']['harmonies']
    ['barnes, clive.txt']
    '''

    associates = get_words_by_root(corpus, word, check_identities)

    assert(associates), 'No such word or associated words in corpus.'

    #see which words are associated with the root 
    print(f'Associated words: {associates}')
    
    by_associate = {}

    for associate in associates:
        
        #Include the following line if you want empty dictionaries for words with no hits.
        #by_associate[associate] = by_associate.get(associate, {})

        try:
            by_associate[associate] = get_ww_instance(corpus, associate, window, stopword_lists, part_of_speech_tag)
        except AssertionError: continue #no instances of word

    return by_associate

def get_ww_by_field(corpus, word, metadata_field, window=5, stopword_lists=None, part_of_speech_tag=None, check_identities=True): 
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    word (str) : the word to count
    metadata_field (str) : the metadata_field to consider
    window (int) : number of words around the input to consider, default is 5
    stopword_lists (list) : list of custom stopword lists, optional
    part_of_speech_tag (str) : part of speech tag (use base tags), optional
    check_identities (bool) : find word equivalents first, default is True

    Returns
    -------
    a dictionary with the metadata field entries as keys and dictionaries as values
    in the format of what is returned by get_ww_by_root
    
    >>> get_ww_by_field(complete, 'dissonance', 'performance_work', 5, [stopwords_nltk])
    {'circus polka': 
        {'dissonant': 
            {'witty': ['frankenstein, alfred6-2.txt'], 
            'russian': ['frankenstein, alfred6-2.txt'],...}},
    'concerto for piano and winds': 
        {'dissonance': 
            {'speaking': ['downes, olin5.txt'], 
            'manner': ['downes, olin5.txt'],...}}, 
    'concerto for two pianos': 
        {'dissonances': 
            {'lines': ['berger, arthur17-2.txt'], 
            'abstraction': ['berger, arthur17-2.txt'],...}}, 
    'danses concertantes': 
        {'dissonant': 
            {'design': ['1944jones.txt'], 
            'artists': ['1944jones.txt'],...}},...}

    >>> get_ww_by_field(complete, 'dissonance', 'performance_work', 5, [stopwords_nltk])['concerto for piano and winds']['dissonance']
    {'speaking': ['downes, olin5.txt'], 'manner': ['downes, olin5.txt'], 'grinding': ['downes, olin5.txt'], 'broadly': ['downes, olin5.txt'], 'fine': ['downes, olin5.txt']}
    '''

    by_value = {}
    
    values = corpus.get_field_vals(metadata_field)  
    #omit the documents whose metadata entry is blank
    if '' in values: values.remove('')
    
    for value in values:
        #create subcorpuses for each entry of metadata field
        current_subcorpus = corpus.subcorpus(metadata_field, value)  

        #Include the following line if you want empty dictionaries for metadata values with no hits.
        #by_value[value] = by_value.get(value, {})

        try:
            by_value[value] = get_ww_by_root(current_subcorpus, word, window, stopword_lists, part_of_speech_tag, check_identities)
        except AssertionError: continue #no instances of words

    return by_value

# sentiment analyses

def get_sentiment_by_document(corpus, filename, word=None, check_identities=True):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    filename (str) : filename associated with a document in the corpus
    word (str) : word to search within the document, optional
    check_identities (bool) : find word equivalents first, default is True
    
    Returns
    -------
    a dictionary of sentence number (from 1 to n, the total number of sentences) as key
    and compound score of that sentence as values

    >>> get_sentiment_by_document(complete, 'laloy, louis.txt')
    {1: 0.0, 2: 0.6369, 3: -0.5994, 4: 0.9382, 5: 0.4939, 6: 0.0, 7: 0.7105, 8: 0.7357}

    >>> get_sentiment_by_document(complete, 'lafond, martine.txt', 'stravinsky')
    {4: 0.8587, 16: -0.0258}
    '''

    #normalize input
    if word == 'None': word = eval(word)

    scores = {}
    
    sentences = get_sentences(corpus, filename) 
    count = 0 #count number of notable sentences, used when there is a word input

    if word: 
        word_list = get_words_by_root(corpus, word, check_identities)
        assert(word_list), 'No such word or associated words in corpus.'
    
    for num in sentences:
        sentence = sentences[num]
        score = sia.polarity_scores(sentence)['compound'] #get the compound score
        if word: #only add sentence that mentions word or its associates
            for word in word_list: 
                if word in sentence.lower(): 
                    scores[num] = score
                    continue #no need to traverse through rest of loop
            try: scores[num] #check if added to dictionary
            except KeyError: continue #if not, move onto next sentence w/o printing any sentence
        else: scores[num] = score #add all sentences 

        if score > 0.5: #print notable sentences
            print(f'Sentence {num} ({sentence}) has a **POSITIVE** compound sentiment score of {score}.')
            count += 1
        if score < -0.5: 
            print(f'Sentence {num} ({sentence}) has a **NEGATIVE** compound sentiment score of {score}.')
            count += 1

    if word: 
        if len(scores)>0: print(f'Associated words were detected within {len(scores)} sentences out of {len(sentences)} in document **{filename}**.')

    if len(scores)>0: print(f'The number of notable sentences is {count}.\n')

    return scores

def graph_sentiment_by_document(corpus, filename, word=None, check_identities=True):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    filename (str) : filename associated with a document in the corpus
    word (str) : word to search within the document, optional
    check_identities (bool) : find word equivalents first, default is True
    
    Returns
    -------
    None.
    '''
    scores = get_sentiment_by_document(corpus, filename, word, check_identities)

    x = list(scores); y = list(scores.values())

    plt.plot(x, y)

    for num in scores:
        score = scores[num]
        if score > 0.5 or score < -0.5:
            plt.annotate(num, (num, score), size=8) #annotate the sentence number if notable

    plt.axhline(y=0.5, color='r')
    plt.axhline(y=-0.5, color='r')

    plt.title(f'Compound scores for {filename} by sentence number; word={word}')
    plt.xlabel("sentence number")
    plt.ylabel("compound sentiment score")
    plt.show()

def get_sentiment_by_field(corpus, metadata_field, word=None, value_list=None, ignore_neutral=True, check_identities=True):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    metadata_field (str) : the metadata_field to consider
    word (str) : word to search within the document, optional
    value_list (list) : list of individual values of the specified metadata field to split corpus by, optional
    ignore_neutral (bool) : whether to omit sentences of compound score 0.0 when tabulating document average, default is True (applicable when a word is specified, can safely ignore otherwise)
    check_identities (bool) : find word equivalents first, default is True (applicable when a word is specified, can safely ignore otherwise)

    Returns
    -------
    a dictionary of metadata values as key and dictionaries as values, in which document
    names are keys and aggregate/average compound scores are values

    >>> get_sentiment_by_field(complete, 'performance_work', None, ['pulcinella', 'rite of spring'])
    {'pulcinella': {'1942jones-2.txt': 0.8161, 'bechert, paul.txt': 0.8195,...}
    'rite of spring': {'box, harold4.txt': 0.908, 'box, harold5-2.txt': 0.3182,...}}
    '''

    #normalize input
    if word == 'None': word = eval(word)
    if type(value_list) == str: 
        try: value_list = eval(value_list)
        except NameError: #not a list
            'Input a valid list of values.'
    if type(ignore_neutral)==str: ignore_neutral = eval(ignore_neutral)

    if value_list: values = value_list #obey input value list
    elif word: #don't bother getting field values in which input word is not mentioned
        values = set()
        wc = get_wordcount_by_field(corpus, metadata_field, word, None, None, check_identities)
        for w in wc:
            field_vals = wc[w].keys()
            values = values.union(set(field_vals))
    else: values = corpus.get_field_vals(metadata_field) #all metadata values are considered

    if '' in values: values.remove('')

    by_value = {}
    for value in values: #iterate over each relevant metadata value
        current_subcorpus = corpus.subcorpus(metadata_field, value)

        for d in current_subcorpus: #iterate over each document
            by_value[value] = by_value.get(value, {}) #set up dictionary for each key, which is a metadata value

            if word: #this might take a lot of time depending on subcorpus size
                scores = get_sentiment_by_document(current_subcorpus, d.filename, word, check_identities)
                
                if ignore_neutral: #omit sentences with compound score of 0.0
                    filtered = ([s for s in scores.values() if s!=0.0])
                    try: 
                        by_value[value][d.filename] = sum(filtered)/len(filtered) #average compound score of document
                    except ZeroDivisionError: #either no sentences with input word or no sentences that have compound scores of 0.0
                        pass
                else: #don't omit sentences with compound score of 0.0
                    try: 
                        by_value[value][d.filename] = sum(scores.values())/len(scores)
                    except ZeroDivisionError: #no sentences with input word
                        pass
            else:
                comp = sia.polarity_scores(d.text)['compound'] #aggregate compound score
                by_value[value][d.filename] = comp

    return by_value

def graph_sentiment_by_field(corpus, metadata_field, word=None, value_list=None, draw_lines=True, ignore_neutral=True, check_identities=True):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    metadata_field (str) : the metadata_field to consider
    word (str) : word to search within the document, optional
    value_list (list) : list of individual values of the specified metadata field to split corpus by, optional
    draw_lines (bool) : whether to connect points on the graph that correspond to the same performance_work, default is True
    ignore_neutral (bool) : whether to omit sentences of compound score 0.0 when tabulating document average, default is True (applicable when a word is specified, can safely ignore otherwise)
    check_identities (bool) : find word equivalents first, default is True (applicable when a word is specified, can safely ignore otherwise)

    Returns
    -------
    None. (A detailed legend is printed.)
    '''

    #normalize input
    if type(draw_lines)==str: draw_lines = eval(draw_lines)
    if word == 'None': word = eval(word)

    plt.rcParams['font.size'] = '8' #set default font size
    plt.rcParams["figure.figsize"] = (16,8) #set default graph size
    plt.axhline(y=0, c='black', linestyle='-') #draw a horizontal line at (0,0)

    by_value = get_sentiment_by_field(corpus, metadata_field, word, value_list, ignore_neutral, check_identities)
 
    pw_values = corpus.get_field_vals('performance_work') #all performance_work values associated with corpus
    if metadata_field != 'performance work': 
        values = sorted(list(by_value.keys())) 
    else: values = pw_values

    #assign colors to each performance_work value
    if len(pw_values)<=20: 
        #colors in list were determined by their maximum distance from one another
        color_options = ['red', 'darkslategray', 'palegreen', 'olive', 'navy',
                'tan', 'maroon', 'orange', 'yellow', 'chartreuse', 
                'plum', 'aqua', 'blue', 'deeppink', 'dodgerblue',
                'salmon', 'seagreen', 'orangered', 'lightskyblue', 'darkviolet']
    else: 
        #colors are not optimized for graphs with more than 20 performance_works
        color_options = list(colors._colors_full_map.values())

    printed = []
    coordinates = {}
    for value in values:
        for filename in by_value[value]: 
            score = by_value[value][filename]

            #find the performance_work associated with the document
            pw = getattr(corpus.get_document('filename', filename), 'performance_work')
            index = pw_values.index(pw) #get index of that performance_work within pw_values
            plt.plot(value, score, '.',  c=color_options[index]) #assign appropriate color and plot point

            #for line drawing
            coordinates[index] = coordinates.get(index, {'x':[], 'y':[]})
            coordinates[index]['x'].append(value)
            coordinates[index]['y'].append(score)
            
            #for detailed legend
            if metadata_field != 'performance_work':
                printed.append((index, f"{index} (performance_work: {pw}). {filename}: {score}, of value '{value}' ({color_options[index]})"))   
            else: 
                printed.append((index, f"{index} (performance_work: {pw}). {filename}: {score} ({color_options[index]})"))

    #draw lines and annotate, if necessary
    for i in coordinates:
        x = coordinates[i]['x']; y = coordinates[i]['y']
        if draw_lines: plt.plot(x, y, c=color_options[i], linewidth=1)
        for j in range(len(x)):
            value = x[j]; score = y[j]
            if len(coordinates)>1: plt.annotate(i, (value, score))#, ha='center')
        
    printed = sorted(printed) #order of ascending index
    for num, string in printed: print(string)

    plt.xticks(np.arange(0, len(values), 1), rotation=90)
    plt.yticks(np.arange(-1, 1.05, 0.05))
    plt.ylabel('compound score')
    
    if not corpus.name: 
        corpus.name = input("\nInput corpus does not have a name. Enter one here to be displayed on the graph: ")
    plt.suptitle(f'Average document compound scores for corpus {corpus.name} by {metadata_field}')
    
    if word: plt.title(f"Filtered by word: '{get_words_by_root(corpus, word)}'")

    plt.tight_layout() 
    plt.show()

# similarity analyses

def get_common_words(corpus, metadata_field, value_list=None, stopword_lists=None, part_of_speech_tag=None):
    '''
    Parameters
    ----------
    corpus (corpus) : the corpus object to split
    metadata_field (str) : the metadata_field to split the corpus by
    value_list (list) : list of individual values of the specified metadata field to split corpus by, optional
    stopword_lists (list) : list of custom stopword lists, optional
    part_of_speech_tag (str) : part of speech tag (use base tags), optional

    Returns
    -------
    a dictionary of words whose values are dictionaries with subcorpus names as keys 
    and individual word counts as values

    >>> get_common_words(complete, 'performance_work', ['pulcinella', 'apollo'], [stopwords_nltk], 'JJ')
    {'melodious': {'pulcinella': 1, 'apollo': 2}, 
    'unexpected': {'pulcinella': 1, 'apollo': 1}, 
    'hard': {'pulcinella': 1, 'apollo': 1}, 
    'old': {'pulcinella': 5, 'apollo': 10},...}
    '''

    #normalize input
    if type(value_list) == str: 
        try: value_list = eval(value_list)
        except NameError: #not a list
            'Input a valid list of values.'

    if value_list: values = value_list #obey input value list
    else: values = corpus.get_field_vals(metadata_field)  

    if '' in values: values.remove('')

    assert(len(values)) > 1, 'The function requires a minimum of two corpuses to compare. Please try again.'
    
    total = []
    for value in values:
        current_subcorpus = corpus.subcorpus(metadata_field, value) #only gets data for necessary values of metadata_field; saves time
        wc = get_wordcount_by_field(current_subcorpus, metadata_field, None, stopword_lists, part_of_speech_tag)
        total.append(wc) #format of {word: {value: count},...} 
        
    temp = total[0] #use the first item as the object of comparison
    for wc in total[1:]:
        for w in wc: #for each key (word) in the second to last dicts
            try: temp[w].update(wc[w]) #if key is in the first dict, add value 
            except KeyError: pass

    common_words = temp.copy()
    length = len(values)
    for w in temp: #remove values that are not/partially present in other
        if len(temp[w])!=length: common_words.pop(w) 
    
    return common_words

def graph_common_words(corpus, metadata_field, value_list=None, stopword_lists=None, part_of_speech_tag=None, limit=None):
    '''
    Parameters
    ----------
    corpus (corpus) : the corpus object to split
    metadata_field (str) : the metadata_field to split the corpus by
    value_list (list) : list of individual values of the specified metadata field to split corpus by, optional
    stopword_lists (list) : list of custom stopword lists, optional
    part_of_speech_tag (str) : part of speech tag (use base tags), optional
    limit (int) : number of common words to include, optional
    
    Returns
    -------
    None.
    '''

    #normalize input
    if type(value_list) == str: 
        try: value_list = eval(value_list)
        except NameError: #not a list
            'Input a valid list of values.'
    if limit: limit = int(limit)

    plt.rcParams['font.size'] = '8' #set default font size

    common_words = get_common_words(corpus, metadata_field, value_list, stopword_lists, part_of_speech_tag)
    assert(len(common_words)>0), f"There are no common terms between the specified values of {metadata_field}."

    if value_list: values = value_list
    else: values = corpus.get_field_vals(metadata_field)  

    pre_order = [] 
    for w in common_words: 
        count = 0
        for value in values: 
            count+=common_words[w][value]
        pre_order.append((w, count))

    #so that the graph starts with words with the highest combined count
    post_order = [t[0] for t in sorted(pre_order, key=lambda x: x[1], reverse=True)]

    if limit: #if placing a limit on how many words to graph
        limit = int(limit) #normalize input
        post_order = post_order[:limit]

    y_max = 0
    for value in values: #plot each metadata value at a time
        x = []; y = []
        for word in post_order:
            x.append(word)
            y.append(common_words[word][value])
        if max(y) > y_max: y_max = max(y) #update what the max overall count is 
        plt.plot(x, y, label=value)

    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, y_max+1, 1.0)) #max is used here to determine appropriate number of y-ticks
    plt.suptitle(f'Frequency vs. common words in corpuses (part-of-speech = {part_of_speech_tag})', fontsize='small')
    plt.title(f'Corpuses involved: {values}', fontsize='xx-small')
    plt.legend(fontsize='xx-small')
    plt.tight_layout()  
    plt.show()

# collocation analyses

def get_collocates(corpus, filename, n, word=None, stopword_lists=None):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    filename (str) : filename associated with a document in the corpus
    n (int) : number of grams
    word (str) : word to search within the document, optional
    stopword_lists (list) : list of custom stopword lists, optional

    Returns
    -------
    a dictionary of words as keys and immediate neighbors as values

    >>> get_collocates(complete, 'antheil, george.txt', 5, 'stravinsky', [stopwords_nltk])
    {'come': ['point'], 'point': ['right', 'contact', 'say'], 'right': ['away'], 
    'away': ['stravinsky'],...}
    '''

    tuples = get_ngrams(corpus, filename, n, word, stopword_lists)
    node_dict = {}

    for t in tuples:
        for i in range(len(t)-1): #ignore last word since there are no words that follow
            node_dict[t[i]] = node_dict.get(t[i], list()) #t[i] represents a word in the tuple
            node_dict[t[i]].append(t[i+1]) #add the immediate neighbor on the right side
    
    return node_dict

def graph_collocates(corpus, n, word, filename_list=None, stopword_lists=None, limit_by=None, layout='shell'):
    '''
    Parameters
    ----------
    corpus (corpus) : a corpus object
    n (int) : number of grams
    word (str) : word to search within each document; becomes the central node
    filename_list (list) : list of filenames to include from corpus, optional
    stopword_lists (list) : list of custom stopword lists, optional
    limit_by (list) : list of two integers where the first specifies limit type 
                    (1: by number affiliated documents or 2: by number of word associations) 
                    and where the second denotes the minimum, optional
    layout (str) : options are 'shell', 'circular', 'planar', 'spring', 'spiral', 'spectral', and 'kamada_kawai', default is 'shell'

    Returns
    -------
    None.
    '''
    
    #normalize input
    n = int(n)
    if type(limit_by) == str: limit_by = eval(limit_by)
    if type(filename_list) == str: filename_list = eval(filename_list)

    assert(n>1), "Please input a number greater than 1."

    print('Node size is determined by number of collocates (successors).')
    print('Node color is determined by proximity to the central node for if len(documents)==1.')
    print('Node color is determined by the number of affiliate documents if len(documents)>1.')

    plt.rcParams["figure.figsize"] = (16,8) #set default graph size

    G = nx.MultiDiGraph() #directed graph

    #step 1. determine appropriate documents to traverse
    if filename_list and len(filename_list)!=0: #empty list means no filenames provided
        assert(type(filename_list)==list), "Please input a list of filename(s)."
        documents = filename_list
    else: #get all filenames
        documents = [document.filename for document in corpus]

    #step 2. traverse (each) document to get its collocates around input word
    #step 3. add nodes, edges, node attributes
    for document in documents:
        try:
            collocates = get_collocates(corpus, document, n, word, stopword_lists)
        except AssertionError: 
            #print(f'Document {document} does not include "{word}"--omitting from analysis.')
            continue #move onto next document

        for node1 in collocates: #node1==preceding term
            if node1 not in G.nodes: #node could have been added already
                G.add_node(node1, documents=[document])#, color=color_options[color_index], shape='o', document=document) #each document has its own color
            else: 
                if document not in G.nodes[node1]["documents"]:
                    G.nodes[node1]["documents"].append(document)

            for node2 in collocates[node1]: #node2==following term
                if node2 not in G.nodes: #node could have been added already
                    G.add_node(node2, documents=[document])#, color=color_options[color_index], shape='o', document=document)
                else: 
                    if document not in G.nodes[node2]["documents"]:
                        G.nodes[node2]["documents"].append(document)

                G.add_edge(node1, node2) #add edge
    
    nodes=G.nodes; edges=G.edges

    scale = math.sqrt(len(nodes))
    center = (0,0)

    node_documents = nx.get_node_attributes(G,'documents')

    #step 4. choose graph layout, configure nodes
    if layout=='shell':
        #to determine nlist; the list of lists that determine node hierarchy
        for i in range(len(nodes)):
            if i == 0: #first level includes all immediate predecessors and successors to central node 
                predecessors = set(G.predecessors(word))
                successors = set(G[word]) #same thing as collocates[word]
                next_level = list(set(predecessors).union(successors))

                nlist = [[word], next_level] #zeroth level is the central node

                nodes_left = set(nodes).difference(set(next_level)) #do not consider nodes added again
                try: nodes_left.remove(word) #remove central node from consideration
                except KeyError: nlist[1].remove(word) #unless it was included in the first level, then remove there

            else: #second through last levels work off of nodes left 
                current_level = nlist[i] 
                next_level = []
                for w in current_level:
                    try: #add successors of each word in current level
                        add = list(set(G[w]).intersection(nodes_left))
                        next_level += add
                        nodes_left = set(nodes_left).difference(set(add)) #update nodes left
                    except KeyError: pass #no successors
                    try: #add predecessors of each word in current level
                        add = list(set(G.predecessors(w)).intersection(nodes_left))
                        next_level += add
                        nodes_left = set(nodes_left).difference(set(add)) #update nodes left
                    except KeyError: pass #no predecessors
                    
                if len(next_level) > 0: #check to see if any nodes were added to this level
                    nlist.append(next_level)
                else: #if empty, the process is finished
                    #print(f'Number of iterations: {i}')
                    break

        #step 4a. organize graph by first document word appears in
        if len(documents)>1:
            for nl in nlist[1:]:
                index = nlist.index(nl)
                order = [node_documents[w][0] for w in nl] #will get first document
                nlist[index] = [nl for order, nl in sorted(zip(order, nl))] #sort by document name

        pos = nx.shell_layout(G, scale=scale, center=center, nlist=nlist)
        pos[word]=center
        
    elif layout=='circular':
        #deviates from the default circular layout graph by creating an inner circle
        #that is made of immediate predecessors/successors of target word
        pos = nx.circular_layout(G, scale=scale, center=center)
        pos[word] = center

        #immediate predecessors or successors to target word
        adj = list(set(G.predecessors(word)).union(set(G[word])))
        if word in adj: adj.remove(word)

        #step 4a. organize graph by first document word appears in
        if len(documents)>1:
            order = [node_documents[w][0] for w in adj]
            adj_by_doc = [adj for order, adj in sorted(zip(order, adj))] #sort by document name
        else: adj_by_doc = adj

        #distribute nodes by color equally around central node
        increment = ((2*math.pi)/len(adj_by_doc))
        for i in range(len(adj_by_doc)):
            pos[adj_by_doc[i]]=((scale/1.5)*math.cos(i*increment), (scale/1.5)*math.sin(i*increment))
        
    #other layouts that are less optimized
    elif layout=='planar':
        pos = nx.planar_layout(G, scale=scale, center=center)
    elif layout=='spring':
        pos = nx.spring_layout(G, scale=scale, center=center, k=0.25)
        pos[word] = center
    elif layout=='spiral':
        pos = nx.spiral_layout(G, scale=scale, center=center)
        pos[word] = center
    elif layout=='spectral':
        pos = nx.spectral_layout(G, scale=scale, center=center)
        pos[word] = center
    elif layout=='kamada_kawai':
        pos = nx.kamada_kawai_layout(G, scale=scale, center=center)
        pos[word] = center
    elif layout=='random':
        pos = nx.random_layout(G, center=center)
        pos[word] = center
    else:
        raise AssertionError('Please enter a valid layout. Options are "shell"(recommended), "circular"(recommended), "planar", "spring", "spiral", "spectral", and "kamada_kawai".') 

    document_counts = [len(node_documents[node]) for node in node_documents]
    document_counts_unique = sorted(set(document_counts)) #list of unique document affiliation counts
    
    #step 5. determine which nodes actually end up on the graph, if applicable
    if limit_by:
        assert isinstance(limit_by, list), "Must input a list in the form of [type, int]."
        assert(limit_by[0]==1 or limit_by[0]==2), "Must select valid limit type, which is 1 (by number of affiliated documents) or 2 (by number of word associations)."
        assert isinstance(limit_by[1], int), "Must input a valid integer to limit by."
        
        if limit_by[0] == 1: #number of affiliated documents
            node_copy = list(nodes)
            for node in node_copy: 
                if len(node_documents[node]) < limit_by[1]: G.remove_node(node)
            
            print(f'\nNumber of documents nodes appear in is one of the following: {document_counts_unique}')
            print(f'Median number of document appearances is {statistics.median(document_counts)}')
            print(f'Mean number of document appearances is {statistics.mean(document_counts)}.')
            print(f'Feel free to alter the limit value based on this finding.')

            limit_type = 'number of affiliated documents'
            
        else: #number of word associations
            connections = []; omit = []
            for node in nodes: 
                predecessors = set(G.predecessors(node))
                successors = set(G[node])
                total = len(predecessors)+len(successors)
                connections.append(total)
                if total < limit_by[1]: omit.append(node)

            for node in omit:
                G.remove_node(node)

            print(f'\nNumber of connections (either direction) per node is one of the following: {sorted(set(connections))} ')
            print(f'Median number of connections (either direction) is {statistics.median(connections)}.')
            print(f'Mean number of connections (either direction) is {statistics.mean(connections)}.')
            print(f'Feel free to alter the limit value based on this finding.')

            limit_type = 'number of word associations'

    else: #dummy values for graph title purposes
        limit_type = None
        limit_by = [None, None]
    
    #step 6. determine node sizes
    sizes = {} #node size is determined by number of collocates (successors)
    for node in nodes:
        try: #will only encounter if node not added to to_remove or not restrict
            scale = 1-(1/len(G[node])) #same thing as collocates[node]
            sizes[node] = sizes.get(node, {'size': 100*10**scale})
        except ZeroDivisionError: #has no successors
            sizes[node] = sizes.get(node, {'size': 25}) #arbitrary
    nx.set_node_attributes(G, sizes)

    #step 7. determine node colors
    legend = set()
    if len(documents)==1: #node color is determined by proximity to the target word

        legend_title = 'distance to central node'
        color_options = sns.mpl_palette("rainbow_r", n) #number of colors to generate is the max length possible from central node to another node

        colors = {}
        for node in nodes: #determine gradation using shortest path
            try: 
                a1 = len(nx.shortest_path(G, node, word))-1
            except nx.NetworkXNoPath: 
                a2 = len(nx.shortest_path(G, word, node))-1
                colors[node] = colors.get(node, {'color': color_options[a2]})
                legend.add((a2+1, color_options[a2]))
                continue

            try: #will only encounter if a1 exists
                a2 = len(nx.shortest_path(G, word, node))-1
                if a1 >= a2: 
                    colors[node] = colors.get(node, {'color': color_options[a2]})
                    legend.add((a2+1, color_options[a2]))
                else: 
                    colors[node] = colors.get(node, {'color': color_options[a1]})
                    legend.add((a1+1, color_options[a1]))
            except nx.NetworkXNoPath: 
                colors[node] = colors.get(node, {'color': color_options[a1]})
                legend.add((a1+1, color_options[a1]))

    else: #node color is determined by number of affiliate documents
        print(f'\nListed below are nodes shown on the graph and the documents in which they appear.')

        legend_title = 'number of affiliate documents'
        maximum = len(document_counts_unique) #represents the total number of unique document affiliation counts
        color_options = sns.mpl_palette("rainbow", maximum) #number of colors to generate is this total number; allows for a distinct enough color palette

        colors = {}
        for node in nodes: 
            index = document_counts_unique.index(len(node_documents[node])) 
            colors[node] = colors.get(node, {'color': color_options[index]})

            legend.add((index+1, color_options[index])) #(document count, affiliated color)

            print(f'The word **{node}** appears in {len(node_documents[node])} documents: {node_documents[node]}')
    nx.set_node_attributes(G, colors)

    #step 8. configure legend, compile lists for node color and size
    legend = sorted(legend)
    handles = [patch.Patch(color=color) for count, color in legend]
    labels = [count for count, color in legend]
    plt.legend(handles, labels, fontsize='xx-small', handlelength=0.75, ncol=2, frameon=False, title=legend_title, title_fontsize='xx-small')

    node_colors = list(nx.get_node_attributes(G,'color').values())
    node_sizes = list(nx.get_node_attributes(G,'size').values())

    #step 9. draw nodes and edges using above information
    nx.draw_networkx_edges(G, pos, width=0.25, arrowsize=5)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.5).set_edgecolor('w') #default alpha=0.5 for visibility purposes
    nx.draw_networkx_labels(G, pos, font_size=5)
    
    #step 10. plot graph, add title
    if not corpus.name: 
        corpus.name = input("\nInput corpus does not have a name. Enter one here to be displayed on the graph: ")

    plt.suptitle(f"Word associations for corpus {corpus.name}, central node: '{word}'", fontsize='small')
    plt.title(f'Nodes filtered by {limit_type}, minimum={limit_by[1]}.', fontsize='x-small')
    plt.show() 
