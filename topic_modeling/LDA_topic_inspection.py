from gensim.models import CoherenceModel, LdaModel
from gensim import corpora, similarities
from gensim.models.callbacks import ConvergenceMetric, CoherenceMetric
from ast import literal_eval
import matplotlib.pyplot as plt
import pyLDAvis.gensim
import pprint as pp
import warnings
import logging
import pymongo
warnings.simplefilter("ignore", DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""
    This script performs inspection and ranking between topics on a given LDA model.

    Requires:
    ----------
    data_path : path where to read and write data
    dictionary : Gensim dictionary [dictionary.dict]
    corpus : Gensim corpus [corpus.mm]
    texts : array of documents of words used to generate corpus
    lda_model : trained lda model 

"""

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["NewsAnalyzer"]

# PIPELINE PARAMETERS
data_path = 'tmp/world/'

# Load texts
with open(data_path+'pp_data_train.txt') as f:
    texts = [literal_eval(line) for line in f]

# Load dictionary
dictionary = corpora.Dictionary.load(data_path+'dictionary.dict')

# Load corpus
corpus = corpora.MmCorpus(data_path+'corpus.mm')

# Load model
lda_model = LdaModel.load(data_path+'LDA_model.lda')

# create model view
p = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(p, data_path+'plots/lda_vis.html')

# rank topics
top_topics = lda_model.top_topics(corpus=corpus, texts=texts, dictionary=dictionary, coherence='c_v', topn=20, processes=-1)
print(top_topics)

'''
# get topics labels
topic_labels = db.topics.distinct('label')
topic_labels.append('other')
topic_counter = {}
for t in topic_labels:
    topic_counter[t] = 0

# count articles by topic labels
for d in corpus:
    probabilities = lda_model.get_document_topics(d, minimum_probability=0.05)
    probabilities = sorted(probabilities, key=lambda x: x[1], reverse=True)
    if len(probabilities) < 1:
        topic_counter['other'] += 1
    else:
        label = None
        for i in range(0, len(probabilities)):
            topid = probabilities[i][0]
            db_topic = db.topics.find_one({'class': 'politics', 'topid': topid})
            if db_topic:
                label = db_topic['label']
                break
        if not label:
            label = 'other'
        topic_counter[label] += 1

topic_counter = sorted(topic_counter.items(), key=lambda kv: kv[1], reverse=True)
pp.pprint(topic_counter)
'''