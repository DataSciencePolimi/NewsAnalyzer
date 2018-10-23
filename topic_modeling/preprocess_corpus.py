import gensim
from gensim.utils import simple_preprocess
# spacy for lemmatization
import spacy
import pickle

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'do', 'say', 'go', 'not', 's', 'tell', 'be'
                   'thing', 'think', 'can', 'could', 'would', 'use', 'have', 'dont', 'make', 'get'])


def preprocess(sentences=None, bigram_model=None, use_bigrams=True, big_min_count=5, big_treshold=100):

    if not sentences:
        return None

    def sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(sentences))

    bigram_mod = None
    if use_bigrams:
        # Build the bigram and trigram models
        if not bigram_model:
            bigram = gensim.models.Phrases(data_words, min_count=big_min_count, threshold=big_treshold)  # higher threshold fewer phrases.
            bigram_mod = gensim.models.phrases.Phraser(bigram)
        else:
            bigram_mod = bigram_model

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def lemmatization(texts, allowed_postags={'NOUN', 'ADJ', 'VERB', 'ADV'}):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words = remove_stopwords(data_words)

    # Form Bigrams
    if use_bigrams:
        data_words = make_bigrams(data_words)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words, allowed_postags={'NOUN', 'ADJ', 'VERB'})
    data_lemmatized = remove_stopwords(data_lemmatized)

    return data_lemmatized, bigram_mod

