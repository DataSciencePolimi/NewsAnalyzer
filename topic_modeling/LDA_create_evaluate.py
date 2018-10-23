from gensim.models import CoherenceModel, LdaModel
from gensim import corpora, similarities
from gensim.models.callbacks import ConvergenceMetric, CoherenceMetric
from ast import literal_eval
import matplotlib.pyplot as plt
import re
import warnings
import logging
warnings.simplefilter("ignore", DeprecationWarning)

"""
    This script trains and evaluates an LDA model given the number of topics.
    Multiple models are trained and the optimal one (by coherence score) is saved.

    Requires:
    ----------
    data_path : path where to read and write data
    dictionary : Gensim dictionary [dictionary.dict]
    corpus : Gensim corpus [corpus.mm]
    texts: array of documents of words used to generate corpus
    topic_number: number of LDA topics
    iterations: number of models to evaluate

"""

# PIPELINE PARAMETERS
data_path = 'tmp/all_32k/'
n_models = 1

# MODEL PARAMETERS:
topic_number = 300
chunksize = 10200
epochs = 20
model_eval_every = 4
max_iterations = 200
alpha = 'auto'
beta = topic_number / 2000

# clear log file
with open('gensim.log', 'w'):
    pass

# Enable logging
logging.basicConfig(filename='gensim.log', format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

# Load texts
with open(data_path+'pp_data_train.txt') as f:
    texts = [literal_eval(line) for line in f]

# Load dictionary
dictionary = corpora.Dictionary.load(data_path+'dictionary.dict')
# Load corpus
corpus = corpora.MmCorpus(data_path+'corpus.mm')


def parse_log_file():
    """
        Function to parse gensim log file and plot training statistics
    """

    print('Evaluating training metrics')
    p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
    matches = [p.findall(l) for l in open('gensim.log')]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    tuples = tuples[2:]
    perplexity = [float(t[1]) for t in tuples]
    liklihood = [float(t[0]) for t in tuples]
    iter = list(range(2, len(tuples) + 2, 1))
    plt.plot(iter, liklihood)
    plt.ylabel("log liklihood")
    plt.xlabel("epochs")
    plt.title("Topic Model Convergence")
    plt.savefig(data_path+"plots/convergence_liklihood.pdf")
    plt.close()

    plt.plot(iter, perplexity)
    plt.ylabel("perplexity")
    plt.xlabel("epochs")
    plt.title("Topic Model Convergence")
    plt.savefig(data_path+"plots/convergence_perplexity.pdf")
    plt.close()

    p = re.compile("topic diff=(\d+\.\d+)")
    matches = [p.findall(l) for l in open('gensim.log')]
    matches = [m for m in matches if len(m) > 0]
    diff = [float(v[0]) for v in matches]
    iter = list(range(0, len(matches), 1))
    plt.plot(iter, diff)
    plt.ylabel("topic diff")
    plt.xlabel("batch")
    plt.title("Topic Model Convergence")
    plt.savefig(data_path+"plots/topic_diff.pdf")
    plt.close()

    p = re.compile("Coherence estimate: (\d+\.\d+)")
    matches = [p.findall(l) for l in open('gensim.log')]
    matches = [m for m in matches if len(m) > 0]
    coherence = [float(v[0]) for v in matches]
    iter = list(range(0, len(matches), 1))
    plt.plot(iter, coherence)
    plt.ylabel("c_v coherence")
    plt.xlabel("epochs")
    plt.title("Topic Model Convergence")
    plt.savefig(data_path+"plots/convergence_cv.pdf")
    plt.close()

    p = re.compile("Convergence estimate: (\d+\.\d+)")
    matches = [p.findall(l) for l in open('gensim.log')]
    matches = [m for m in matches if len(m) > 0]
    coherence = [float(v[0]) for v in matches]
    iter = list(range(0, len(matches), 1))
    plt.plot(iter, coherence)
    plt.ylabel("convergence metric")
    plt.xlabel("epochs")
    plt.title("Topic Model Convergence")
    plt.savefig(data_path+"plots/convergence_conv.pdf")
    plt.close()


# gensim callbacks to monitor training process
convergence_logger = ConvergenceMetric(logger='shell')
coherence_logger = CoherenceMetric(texts=texts, corpus=corpus, dictionary=dictionary, coherence='c_v', logger='shell')

best_model = None
best_coherence = 0

# this loop selects the best model among the [iterations] models trained
for i in range(0, n_models):
    lm = LdaModel(corpus=corpus, id2word=dictionary,
                  num_topics=topic_number,
                  chunksize=chunksize,
                  passes=epochs,
                  eval_every=model_eval_every,
                  # minimum_probability=0.0,
                  iterations=max_iterations,
                  alpha=alpha,
                  eta=beta,
                  # callbacks=[coherence_logger, convergence_logger]
                  )
    best_model = lm
    parse_log_file()

    # cm = CoherenceModel(model=lm, corpus=corpus, texts=texts, dictionary=dictionary, coherence='c_v')
    # c_v = cm.get_coherence()
    # print(c_v)
'''
    if c_v > best_coherence:
        best_coherence = c_v
        best_model = lm
        parse_log_file()
        # clear log file
        with open('gensim.log', 'w'):
            pass
    '''
# print('Best coherence found: ' + str(best_coherence))

# store best model
best_model.save(data_path+'LDA_model.lda')
# print(best_model.top_topics(corpus=corpus, texts=texts, dictionary=dictionary, coherence='c_v'))





