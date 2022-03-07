import nltk
import collections

from data_management.corpus import *
from data_management.document import *
from data_management.stopwords import *
from data_management.conditions import *
from analysis.gender_frequency import get_counts_by_pos, get_count_words
from analysis.instance_distance import words_instance_dist
from analysis.modeling import *

complete = Corpus('data/complete','data/complete/_metadata.csv')
rake = corpus_to_list(complete.subcorpus('performance_work', "rake's progress"))
pulcinella = corpus_to_list(complete.subcorpus('performance_work', 'pulcinella'))
danses = corpus_to_list(complete.subcorpus('performance_work', 'danses concertantes'))
firebird = corpus_to_list(complete.subcorpus('performance_work', 'firebird'))
apollo = corpus_to_list(complete.subcorpus('performance_work', 'apollo'))
concerto = corpus_to_list(complete.subcorpus('performance_work', 'concerto for piano and winds'))

#silhouette_score(rake, 1000, stopwords_nltk+stopwords_general+stopwords_rake) #k=15
#nmf(rake, 1000, 15, 10, stopwords_nltk+stopwords_general+stopwords_rake)
#silhouette_score(pulcinella, 1000, stopwords_nltk+stopwords_general+stopwords_pul) #k=15
#nmf(pulcinella, 1000, 15, 10, stopwords_nltk+stopwords_general+stopwords_pul)
#silhouette_score(danses, 1000, stopwords_nltk+stopwords_general) 
#nmf(danses, 1000, 15, 10, stopwords_nltk+stopwords_general)
#silhouette_score(firebird, 1000, stopwords_nltk+stopwords_general) 
#nmf(firebird, 1000, 15, 10, stopwords_nltk+stopwords_general)
#silhouette_score(apollo, 1000, stopwords_nltk+stopwords_general+stopwords_apollo) 
#nmf(apollo, 1000, 15, 10, stopwords_nltk+stopwords_general+stopwords_apollo)
#silhouette_score(concerto, 1000, stopwords_nltk+stopwords_general+stopwords_cpwi) 
#nmf(concerto, 1000, 15, 10, stopwords_nltk+stopwords_general+stopwords_cpwi)

complete_list = corpus_to_list(complete)

tc_rake = [('C', 'C'),('P', 'S'),('L', 'L'),('C', 'A'),('L', 'A'),('P', 'S'),('L', 'L'),('C', 'C'),('C', 'S'),('C', 'C'),('C', 'C'),('C', 'L'),('C', 'C'),('C', 'C'),('C', 'C')]
tc_pul = [('C', 'C'),('P', 'P'),('C', 'S'),('L', 'C'),('L', 'L'),('C', 'A'),('A', 'A'),('S', 'A'),('C', 'C'),('C', 'C'),('A', 'A'),('C', 'A'),('A', 'A'),('C', 'C'),('C', 'C')]
tc_danses = [('L', 'S'),('C', 'C'),('P', 'C'),('S','C'),('C', 'C'),('S', 'S'),('C', 'C'),('L', 'C'),('C', 'C'),('C', 'C'),('S', 'C'),('L', 'L'),('L', 'C'),('S', 'S'),('L', 'A')]
tc_fireb = [('L', 'C'),('C', 'L'),('C', 'P'),('C', 'L'),('S', 'C'),('C', 'C'),('C', 'S'),('C', 'C'),('S', 'L'),('C', 'S'),('P', 'S'),('C', 'P'),('C', 'C'),('S', 'C'),('L', 'L')]
tc_apollo = [('C', 'C'),('C', 'C'),('S', 'C'),('C', 'A'),('P', 'C'),('C', 'C'),('A', 'C'),('S', 'C'),('S', 'S'),('L', 'S'),('L', 'L'),('C', 'C'),('A', 'S'),('C', 'C'),('A', 'A')]
tc_concerto = [('A', 'A'),('S', 'C'),('C', 'C'),('A', 'P'),('L', 'C'),('C', 'C'),('C', 'C'),('A', 'C'),('P', 'S'),('L', 'C'),('C', 'A'),('A', 'A'),('C', 'A'),('A', 'A'),('S', 'C')]
topic_categories = tc_rake+tc_pul+tc_danses+tc_fireb+tc_apollo+tc_concerto

topic_by_document_rake = topic_by_document(rake, 1000, 15, stopwords_nltk+stopwords_general+stopwords_rake)
topic_by_document_pul = topic_by_document(rake, 1000, 15, stopwords_nltk+stopwords_general+stopwords_pul)
topic_by_document_dans = topic_by_document(rake, 1000, 15, stopwords_nltk+stopwords_general)
topic_by_document_fire = topic_by_document(rake, 1000, 15, stopwords_nltk+stopwords_general)
topic_by_document_apol = topic_by_document(rake, 1000, 15, stopwords_nltk+stopwords_general+stopwords_apollo)
topic_by_document_cpwi = topic_by_document(rake, 1000, 15, stopwords_nltk+stopwords_general+stopwords_cpwi)

topic_by_category_rake = {0: 'C', 1: 'P', 2: 'L', 3: 'C', 4: 'L', 5: 'P', 6: 'L', 7: 'C', 8: 'C', 9: 'C', 10: 'C', 11: 'C', 12: 'C', 13: 'C', 14: 'C'}
topic_by_category_pul = {0: 'C', 1: 'P', 2: 'C', 3: 'L', 4: 'L', 5: 'C', 6: 'A', 7: 'S', 8: 'C', 9: 'C', 10: 'A', 11: 'C', 12: 'A', 13: 'C', 14: 'C'}
topic_by_category_dans = {0: 'L', 1: 'C', 2: 'P', 3: 'S', 4: 'C', 5: 'S', 6: 'C', 7: 'L', 8: 'C', 9: 'C', 10: 'S', 11: 'L', 12: 'L', 13: 'S', 14: 'L'}
topic_by_category_fire = {0: 'L', 1: 'C', 2: 'C', 3: 'C', 4: 'S', 5: 'C', 6: 'C', 7: 'C', 8: 'S', 9: 'C', 10: 'P', 11: 'C', 12: 'C', 13: 'S', 14: 'L'}
topic_by_category_apol = {0: 'C', 1: 'C', 2: 'S', 3: 'C', 4: 'P', 5: 'C', 6: 'A', 7: 'S', 8: 'S', 9: 'L', 10: 'L', 11: 'C', 12: 'A', 13: 'C', 14: 'A'}
topic_by_category_cpwi = {0: 'A', 1: 'S', 2: 'C', 3: 'A', 4: 'L', 5: 'C', 6: 'C', 7: 'A', 8: 'P', 9: 'L', 10: 'C', 11: 'A', 12: 'C', 13: 'A', 14: 'S'}

graph_document_categories(topic_by_document_rake, topic_by_category_rake, f"Rake's Progress, n={len(rake)}")
graph_document_categories(topic_by_document_pul, topic_by_category_pul, f"Pulcinella, n={len(pulcinella)}")
graph_document_categories(topic_by_document_dans, topic_by_category_dans, f"Danses Concertantes, n={len(danses)}")
graph_document_categories(topic_by_document_fire, topic_by_category_fire, f"Firebird, n={len(firebird)}")
graph_document_categories(topic_by_document_apol, topic_by_category_apol, f"Apollo, n={len(apollo)}")
graph_document_categories(topic_by_document_cpwi, topic_by_category_cpwi, f"Concerto for Piano and Winds, n={len(concerto)}")