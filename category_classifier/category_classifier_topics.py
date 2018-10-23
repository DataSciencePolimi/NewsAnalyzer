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
from gensim import corpora
from ast import literal_eval
warnings.simplefilter("ignore", DeprecationWarning)

"""
    This script create and evaluates different classification models
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
    models

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


def create_model(use_preproc_data=True):
    LDA_models_paths = ['50/LDA_model.lda', '100/LDA_model.lda', '200/LDA_model.lda', '300/LDA_model.lda']
    LDA_models_size = [50, 100, 200, 300]
    data_path = '../topic_modeling/tmp/all_32k/'

    # load the dataset
    if use_preproc_data:
        with open(data_path + 'pp_data_train.txt') as f:
            texts_train = [literal_eval(line) for line in f]

        with open(data_path + 'pp_data_test.txt') as f:
            texts_test = [literal_eval(line) for line in f]

        y_train = pickle.load(open(data_path+'target_train.sav', 'rb'))
        y_test = pickle.load(open(data_path + 'target_test.sav', 'rb'))

    else:
        # Load categories from the training set
        classes = ['world', 'politics', 'business', 'sports',
                   'entertainment/art',
                   'science/technology/health',
                   'national/local',
                   'style/food/travel'
                   ]

        EXCLUDE_SOURCES = ['The Guardian',
                           'CBC News',
                           'Reuters',
                           'BBC',
                           'The Globe and Mail',
                           'New York Daily News',
                           'Los Angeles Times',
                           'News Channel 8'
                           # 'USA Today'
                           ]

        mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = mongo_client["NewsAnalyzer"]

        # load the dataset
        articles = []
        for cat in classes:
            for aa in list(db.articles.aggregate([{"$match":
                                                       {"category_aggregate": cat,
                                                        "ground_truth": True, "source_name": {"$nin": EXCLUDE_SOURCES}
                                                        }},
                                                  {"$sample": {"size": 10000}}])):
                articles.append(aa)

        bigram_model = pickle.load(open(data_path+'bigram_model.sav', 'rb'))
        texts_train = [t['text'] for t in articles]
        targets = [t['category_aggregate'] for t in articles]
        texts_train, bigram_model = preprocessor.preprocess(sentences=texts_train, bigram_model=bigram_model)
        le = preprocessing.LabelEncoder()
        le.fit(classes)
        y_encoded = le.transform(targets)
        texts_train, texts_test, y_train, y_test = train_test_split(texts_train, y_encoded, test_size=0.05, random_state=10)

    dictionary = corpora.Dictionary.load(data_path + 'dictionary.dict')
    lda_models = []
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

    for d in corpus_train:
        topic_vector = []
        for i in range(0, len(LDA_models_size)):
            dst = np.zeros(LDA_models_size[i])
            for p in lda_models[i].get_document_topics(d, minimum_probability=0.05):
                dst[p[0]] = p[1]
            topic_vector.extend(dst)
        X_train.append(topic_vector)

    for d in corpus_test:
        topic_vector = []
        for i in range(0, len(LDA_models_size)):
            dst = np.zeros(LDA_models_size[i])
            for p in lda_models[i].get_document_topics(d, minimum_probability=0.05):
                dst[p[0]] = p[1]
            topic_vector.extend(dst)
        X_test.append(topic_vector)


    # training Naive Bayes model
    clf_NB = MultinomialNB(class_prior=None, fit_prior=True)
    clf_NB.fit(X_train, y_train)
    train_acc_nb = clf_NB.score(X_train, y_train)
    test_acc_nb = clf_NB.score(X_test, y_test)
    print('')
    print('NAIVE BAYES ---------------------------------------------')
    print('Train score: ' + str(train_acc_nb))
    print('Test score: ' + str(test_acc_nb))

    # training SGD classifiers
    from sklearn.linear_model import SGDClassifier
    clf_SVM = SGDClassifier(n_jobs=-1, max_iter=5000, tol=0.0001, learning_rate='optimal', loss='hinge')
    clf_SVM.fit(X_train, y_train)
    train_acc_svm = clf_SVM.score(X_train, y_train)
    test_acc_svm = clf_SVM.score(X_test, y_test)
    print('')
    print('LINEAR SVM ----------------------------------------------')
    print('Train score: ' + str(train_acc_svm))
    print('Test score: ' + str(test_acc_svm))

    clf_log_reg = SGDClassifier(n_jobs=-1, max_iter=5000, tol=0.0001, learning_rate='optimal', loss='modified_huber')
    clf_log_reg.fit(X_train, y_train)
    train_acc_log = clf_log_reg.score(X_train, y_train)
    test_acc_log = clf_log_reg.score(X_test, y_test)
    cm = confusion_matrix(y_test, clf_log_reg.predict(X_test))
    print('')
    print('LOGISTIC REGRESSION -------------------------------------')
    print('Train score: ' + str(train_acc_log))
    print('Test score: ' + str(test_acc_log))

    print('')
    print('--------------------------------------')
    print('')
    print('COUNFUSION MATRIX FOR LOGISTIC REGRESSION:')
    print('')
    print(cm)
    print('')
    print('')

    from sklearn.ensemble import RandomForestClassifier
    clf_random_for = RandomForestClassifier(n_jobs=-1, n_estimators=20, max_depth=None, random_state=0)
    clf_random_for.fit(X_train, y_train)
    train_acc_rf = clf_random_for.score(X_train, y_train)
    test_acc_rf = clf_random_for.score(X_test, y_test)
    print('')
    print('RANDOM FOREST -------------------------------------------')
    print('Train score: ' + str(train_acc_rf))
    print('Test score: ' + str(test_acc_rf))


    # save model, vectors and label encoding to disk
    # save_model(clf_log_reg, count_vect, le)


def main():
    create_model(use_preproc_data=False)


if __name__ == "__main__":
    timeStart = datetime.datetime.now()
    main()
    timeEnd = datetime.datetime.now()
    delta = timeEnd - timeStart
    print('Executed in ' + str(int(delta.total_seconds())) + 's')
