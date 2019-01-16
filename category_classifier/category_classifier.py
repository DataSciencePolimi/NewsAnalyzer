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
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
import pickle
import datetime
import warnings
import logging
warnings.simplefilter("ignore", DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""
    This script create and evaluates different classification models
    for the article class problem.
    
    FEATURES: keywords (from newspaper3k api)
    
    MODELS:
    1) Naive bayes
    2) Linear SVM
    3) Logistic regression
    4) Random Forest

    Requires:
    ----------
    articles in mongodb collection

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


def create_model():
    ARTICLES_PER_CATEG0RY = 10000
    TEST_SIZE = 0.05

    # #############################################################################
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
                                                "ground_truth": True, "source_name": { "$nin": EXCLUDE_SOURCES}
                                                }},
                                              {"$sample": {"size": ARTICLES_PER_CATEG0RY}}])):
            articles.append(aa)

    print('')
    print('')
    print('Training examples: ' + str(len(articles)))
    print('Number of classes: ' + str(len(classes)))
    print('Test set size: ' + str(TEST_SIZE*100) + '%')

    print('')
    print('Pre-processing features...')

    # X -> features, y -> label
    X = []
    y = []

    # create features as concatenation of keywords and tags, labels as category
    for a in articles:
        if a['category_aggregate'] in classes:

            kw_concat = ' '.join(a['keywords'])
            if len(a['tags']) > 2:
                kw_concat = kw_concat + ' ' + ' '.join(a['tags'])

            X.append(kw_concat)
            y.append(a['category_aggregate'])

    # encode labels into integers
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    y_encoded = le.transform(y)

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_SIZE, random_state=10)

    # count vectorize features
    count_vect = CountVectorizer(max_df=0.5, min_df=2, max_features=None, binary=False)
    X_train_countvect = count_vect.fit_transform(X_train)
    X_test_countvect = count_vect.transform(X_test)
    print('COUNTVECTOR, Feature vector size: ' + str(len(count_vect.vocabulary_)))
    print('Training models...')

    # training Naive Bayes model
    clf_NB = MultinomialNB(class_prior=None, fit_prior=True)
    clf_NB.fit(X_train_countvect, y_train)
    train_acc_nb = clf_NB.score(X_train_countvect, y_train)
    test_acc_nb = clf_NB.score(X_test_countvect, y_test)
    print('')
    print('NAIVE BAYES ---------------------------------------------')
    print('Train score: ' + str(train_acc_nb))
    print('Test score: ' + str(test_acc_nb))

    # training SGD classifiers
    from sklearn.linear_model import SGDClassifier
    clf_SVM = SGDClassifier(n_jobs=-1, max_iter=5000, tol=0.0001, learning_rate='optimal', loss='hinge')
    clf_SVM.fit(X_train_countvect, y_train)
    train_acc_svm = clf_SVM.score(X_train_countvect, y_train)
    test_acc_svm = clf_SVM.score(X_test_countvect, y_test)
    print('')
    print('LINEAR SVM ----------------------------------------------')
    print('Train score: ' + str(train_acc_svm))
    print('Test score: ' + str(test_acc_svm))

    clf_log_reg = SGDClassifier(n_jobs=-1, max_iter=5000, tol=0.0001, learning_rate='optimal', loss='modified_huber')
    clf_log_reg.fit(X_train_countvect, y_train)
    train_acc_log = clf_log_reg.score(X_train_countvect, y_train)
    test_acc_log = clf_log_reg.score(X_test_countvect, y_test)
    cm = confusion_matrix(y_test, clf_log_reg.predict(X_test_countvect))
    print('')
    print('LOGISTIC REGRESSION -------------------------------------')
    print('Train score: ' + str(train_acc_log))
    print('Test score: ' + str(test_acc_log))

    print('')
    print('--------------------------------------')
    print('CLASS ENCODING')
    i = 0
    for c in classes:
        print(str(i) + ': ' + c)
        i += 1
    print('')
    print('COUNFUSION MATRIX FOR LOGISTIC REGRESSION:')
    print('')
    print(cm)
    print('')
    print('')

    from sklearn.ensemble import RandomForestClassifier
    clf_random_for = RandomForestClassifier(n_jobs=-1, criterion='entropy', n_estimators=20, max_depth=250, random_state=0, min_samples_leaf=1, verbose=True)
    clf_random_for.fit(X_train_countvect, y_train)
    train_acc_rf = clf_random_for.score(X_train_countvect, y_train)
    test_acc_rf = clf_random_for.score(X_test_countvect, y_test)
    print('')
    print('RANDOM FOREST -------------------------------------------')
    print('Train score: ' + str(train_acc_rf))
    print('Test score: ' + str(test_acc_rf))


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
