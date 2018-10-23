from gensim.models import CoherenceModel, LdaModel
from gensim import corpora
import matplotlib.pyplot as plt
import pickle
import warnings
import logging
import numpy as np
from ast import literal_eval

"""
    This script estimates the optimal number of topics to train an LDA model
    based on a given corpus and dictionary

    Requires:
    ----------
    data_path : path where to read and write data
    dictionary : Gensim dictionary [dictionary.dict]
    corpus : Gensim corpus [corpus.mm]
    texts: array of documents of words used to generate corpus
    iterations: number of iterations of the whole process
    
"""

warnings.simplefilter("ignore", DeprecationWarning)
np.errstate(invalid='ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# PARAMETERS
iterations = 1
data_path = 'tmp/all_32k/'
dictionary = corpora.Dictionary.load(data_path+'dictionary.dict')
corpus = corpora.MmCorpus(data_path+'corpus.mm')
with open(data_path+'pp_data_train.txt') as f:
    texts = [literal_eval(line) for line in f]

# TOPICS GRID
minimum_topics = 100
maximum_topics = 550
step_size = 100


def evaluate_graph(dict=None, corp=None, texts=None, min_topics=10, limit=100, step=50, coh_measure='c_v', iteration=0):
    """
    Function to display num_topics - LDA graph using c_v coherence

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    min_topics: starting topic number
    step: step over topic number

    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """

    c_v = []
    lm_list = []
    for num_topics in range(min_topics, limit, step):
        # LDA params
        chunksize = 10000
        epochs = 1
        model_eval_every = 4
        max_iterations = 50
        alpha_val = 'auto'
        beta = num_topics / 2000

        lm = LdaModel(corpus=corp, num_topics=num_topics, id2word=dict,
                      chunksize=chunksize, passes=epochs, iterations=max_iterations,
                      eval_every=model_eval_every, alpha=alpha_val, eta=beta)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, corpus=corpus, texts=texts, dictionary=dict, coherence=coh_measure)
        coh = cm.get_coherence()
        c_v.append(coh)
        print('Topic count: ' + str(num_topics) + ' coherence: ' + str(coh))

    # Save graph
    x = range(min_topics, limit, step)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.savefig(data_path+"plots/topic_sel_coherence_iter" + str(iteration)+".pdf")
    plt.close()

    return lm_list, c_v


# repeat the process [iterations] times to minimize stochasticity of LDA
for k in range(0, iterations):
    models, c_v = evaluate_graph(dict=dictionary, corp=corpus, texts=texts,
                                 min_topics=minimum_topics, limit=maximum_topics+1,
                                 step=step_size, coh_measure='c_v', iteration=k)
    c_v_values = [{'topics': x} for x in range(minimum_topics, maximum_topics+1, step_size)]
    for i in range(0, len(c_v)):
        c_v_values[i]['c_v'] = c_v[i]
    # store c_v values
    pickle.dump(c_v_values, open(data_path+'plots/cv_values_iter'+str(k)+'.sav', 'wb'))




