import os
from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def transform(n_topics=100, similarity_size=100):
    if os.path.exists("tmp/dictionary.dict") and os.path.exists("tmp/corpus.mm"):
        dictionary = corpora.Dictionary.load('tmp/dictionary.dict')
        corpus = corpora.MmCorpus('tmp/corpus.mm')
    else:
        print("Missing dictionary / corpus files.")
        return

    print('Transforming corpus to Tfidf')
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    tfidf.save('tmp/model.tfidf')

    print('Transforming corpus to LSI Space, num topics: ' + str(n_topics))
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics) # initialize an LSI transformation
    corpus_lsi = lsi[corpus_tfidf]
    lsi.save('tmp/model.lsi')

    print('Transforming corpus to LDA Space, num topics: ' + str(n_topics))
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics, eval_every=5,
                          chunksize=10000, passes=10, iterations=50, update_every=1, minimum_probability=0.0, alpha='auto', eta='auto')  # initialize an LDA transformation
    corpus_lda = lda[corpus_tfidf]
    lda.save('tmp/model.lda')

    print('Generating similarity indexes')
    index_lsi = similarities.MatrixSimilarity(corpus_lsi, num_features=n_topics, num_best=similarity_size)
    index_lsi.save('tmp/sim_index_lsi.index') # index = similarities.MatrixSimilarity.load('tmp/q_index.index')

    index_lda = similarities.MatrixSimilarity(corpus_lda, num_features=n_topics, num_best=similarity_size)
    index_lda.save('tmp/sim_index_lda.index')

    print('Done')

