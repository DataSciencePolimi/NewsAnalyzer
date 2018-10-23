import topic_modeling.preprocess_corpus as preprocessor
import gensim.corpora as corpora
import os
import logging
import pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def create(path='tmp/'):
    if not os.path.exists(path+"raw_data_train.txt"):
        print('Missing raw data.')
        return

    texts = []
    with open(path+'raw_data_train.txt', 'r') as filehandle:
        filecontents = filehandle.readlines()
        for line in filecontents:
            # remove linebreak which is the last character of the string
            current_place = line[:-1]
            # add item to the list
            texts.append(current_place)

    texts_test = []
    with open(path+'raw_data_test.txt', 'r') as filehandle:
        filecontents = filehandle.readlines()
        for line in filecontents:
            # remove linebreak which is the last character of the string
            current_place = line[:-1]
            # add item to the list
            texts_test.append(current_place)

    print('Preprocessing texts...')
    texts, bigram_mod = preprocessor.preprocess(texts, use_bigrams=True)
    pickle.dump(bigram_mod, open(path+'bigram_model.sav', 'wb'))
    with open(path+'pp_data_train.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % a for a in texts)

    texts_test, bigram_mod = preprocessor.preprocess(texts_test, use_bigrams=True, bigram_model=bigram_mod)
    with open(path+'pp_data_test.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % a for a in texts_test)

    print('Creating dictionary...')
    dictionary = corpora.Dictionary(texts)
    # Filter out words that occur less than 10 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    dictionary.save(path+'dictionary.dict')  # store the dictionary, for future reference

    print('Serializing corpus...')
    corpora.MmCorpus.serialize(path+'corpus.mm', [dictionary.doc2bow(t) for t in texts])

