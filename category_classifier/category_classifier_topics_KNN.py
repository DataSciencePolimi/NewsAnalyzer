# importing necessary libraries
import pymongo
from math import exp
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
import datetime
import warnings
import topic_modeling.preprocess_corpus as preprocessor
from gensim.models import LdaModel
from gensim import corpora, similarities
from ast import literal_eval
warnings.simplefilter("ignore", DeprecationWarning)

"""
    This script create and evaluates a KNN model
    for the article class problem.

    FEATURES: topics

    MODELS:
    1) Naive bayes
    2) Linear SVM
    3) Logistic regression
    4) Random Forest

    Requires:
    ----------
    LDA models
    AND
    articles in mongodb collection to be preprocessed
    OR 
    preprocessed corpus (train and test)

    Returns:
    ----------
    model
"""

# PARAMETERS
# minimum probability to accept prediction
classifier_threshold = 0.5
lemmatizer = WordNetLemmatizer()


def save_model(classifier, vect, labelencoder):
    print("...... saving model as pickle file ......")
    pickle.dump(classifier, open('../models/classifier.sav', 'wb'))
    pickle.dump(vect, open('../models/countvect.sav', 'wb'))
    # pickle.dump(tfidfvect, open('../models/tfidfvect.sav', 'wb'))
    pickle.dump(labelencoder, open('../models/labelencoder.sav', 'wb'))
    print("Done.")


# predicts category given a LIST of keywords
# output:
#   {
#       'class': category_prediction,
#       'probability': confidence value (0,1)
#   }
#   if probability < classifier_threshold returns None

def predict(keywords, models=None):
    if not models:
        # load the model from disk
        model = pickle.load(open('category_classifier/models/classifier.sav', 'rb'))
        countvect = pickle.load(open('category_classifier/models/countvect.sav', 'rb'))
        # tfidfvect = pickle.load(open('category_classifier/models/tfidfvect.sav', 'rb'))
        labelencoder = pickle.load(open('category_classifier/models/labelencoder.sav', 'rb'))
    else:
        try:
            model = models['model']
            countvect = models['countvect']
            # tfidfvect = models['tfidfvect']
            labelencoder = models['labelencoder']
        except KeyError:
            print('Bad model provided')
            return None

    kw_concat = ' '.join(keywords)
    X = [kw_concat]
    x_count = countvect.transform(X)

    prediction = model.predict_log_proba(x_count)
    maxprob = max(prediction[0])
    if exp(maxprob) > classifier_threshold:
        for i in range(0, len(prediction[0])):
            if prediction[0][i] == maxprob:
                return {'class': labelencoder.inverse_transform(i), 'probability': exp(maxprob)}
    return None


def get_aggregated_category(category):
    if category in ['national', 'local']:
        return 'national/local'
    elif category in ['entertainment', 'art']:
        return 'entertainment/art'
    elif category in ['science', 'technology', 'health']:
        return 'science/technology/health'
    elif category in ['style', 'food', 'travel']:
        return 'style/food/travel'
    else:
        return category


def transform_keywords_array(key_arr):
    text = transform_keywords(' '.join(key_arr))
    return text.split(' ')


def transform_keywords(text):
    pos_dict = {'NOUN': 'n', 'FW': 'n', 'NN': 'n', 'NNP': 'n', 'NNPS': 'n', 'NNS': 'n',
                'ADJ': 'a', 'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
                'VERB': 'v', 'VB': 'v', 'VBD': 'v', 'VBG': 'v',
                'VBN': 'v', 'VBP': 'v', 'VBZ': 'v'}

    tokens = word_tokenize(text)  # Generate list of tokens
    tokens_pos = pos_tag(tokens)

    stemmed_kw = []

    for kw in tokens_pos:
        word = kw[0].lower()
        pos = kw[1]
        if pos not in pos_dict:
            continue
        pos = pos_dict[pos]
        stemmed_kw.append(lemmatizer.lemmatize(word, pos))

    return ' '.join(stemmed_kw)


def create_model():
    LDA_models_paths = ['50/LDA_model.lda', '100/LDA_model.lda', '200/LDA_model.lda', '300/LDA_model.lda']
    LDA_models_size = [50, 100, 200, 300]
    # load the dataset
    data_path = '../topic_modeling/tmp/all_32k/'
    with open(data_path + 'pp_data_train.txt') as f:
        texts_train = [literal_eval(line) for line in f]

    with open(data_path + 'pp_data_test.txt') as f:
        texts_test = [literal_eval(line) for line in f]

    y_train = pickle.load(open(data_path+'target_train.sav', 'rb'))
    y_test = pickle.load(open(data_path + 'target_test.sav', 'rb'))

    dictionary = corpora.Dictionary.load(data_path + 'dictionary.dict')
    lda_models = []
    similarity_indexes = []
    for m in LDA_models_paths:
        lda_models.append(LdaModel.load(data_path + 'lda_models/' + m))

    print('')
    print('')
    print('Training examples: ' + str(len(texts_train)))
    print('Test examples: ' + str(len(y_test)))
    print('')
    print('Pre-processing features...')

    corpus_train = [dictionary.doc2bow(d) for d in texts_train]
    corpus_test = [dictionary.doc2bow(d) for d in texts_test]

    X_train = []
    X_test = []

    for i in range(0, len(LDA_models_size)):
        lda_model = lda_models[i]
        X_train.append(lda_model[corpus_train])
        X_test.append(lda_model[corpus_test])
        similarity_indexes.append(similarities.MatrixSimilarity(X_train[i], num_features=LDA_models_size[i], num_best=300))

    def evaluate_model(test_vectors, sim_index, n_voting):
        positive_count = 0
        errors = 0

        for i in range(0, len(test_vectors)):
            sims = sim_index[test_vectors[i]]  # perform a similarity query against the corpus
            sims = sorted(sims, key=lambda item: -item[1])
            sims = sims[:n_voting]

            voting_pool = [0] * 8
            for s in sims:
                doc_index = s[0]
                doc_category = y_train[doc_index]
                voting_pool[doc_category] += 1

            category_votes = []
            for j in range(0, len(voting_pool)):
                category_votes.append((j, voting_pool[j]))

            category_votes = sorted(category_votes, key=lambda item: -item[1])
            if category_votes[0][0] == y_test[i]:
                positive_count += 1
            else:
                errors += 1
        return positive_count, errors

    for v in [5, 10, 20, 30, 50, 100, 200, 300]:
        for s in range(0, len(similarity_indexes)):
            positive, errors = evaluate_model(X_test[s], similarity_indexes[s], v)
            print('Accuracy KNN voting: ' + str(v) + ' topic model: ' + str(LDA_models_size[s]) + ': ' + str(positive/len(y_test)))

    # save model, vectors and label encoding to disk
    # save_model(clf_log_reg, count_vect, le)


def main():
    create_model()


if __name__ == "__main__":
    timeStart = datetime.datetime.now()
    main()
    timeEnd = datetime.datetime.now()
    delta = timeEnd - timeStart
    print('Executed in ' + str(int(delta.total_seconds())) + 's')
