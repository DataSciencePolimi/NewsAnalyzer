# importing necessary libraries
import pymongo
from math import exp
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import datetime
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# PARAMETERS
# minimum probability to accept prediction
classifier_threshold = 0.4
lemmatizer = WordNetLemmatizer()


def save_model(classifier, countvect, labelencoder):
    print("...... saving model as pickle file ......")
    pickle.dump(classifier, open('../models/classifier.sav', 'wb'))
    pickle.dump(countvect, open('../models/countvect.sav', 'wb'))
#    pickle.dump(tfidfvect, open('../models/tfidfvect.sav', 'wb'))
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
        model = pickle.load(open('../models/classifier.sav', 'rb'))
        countvect = pickle.load(open('../models/countvect.sav', 'rb'))
        tfidfvect = pickle.load(open('../models/tfidfvect.sav', 'rb'))
        labelencoder = pickle.load(open('../models/labelencoder.sav', 'rb'))
    else:
        try:
            model = models['model']
            countvect = models['countvect']
            tfidfvect = models['tfidfvect']
            labelencoder = models['labelencoder']
        except KeyError:
            print('Bad model provided')
            return None

    # kw_concat = transform_keywords(' '.join(keywords))
    kw_concat = ' '.join(keywords)
    X = [kw_concat]
    x_count = countvect.transform(X)
    feature_vect = tfidfvect.transform(x_count)
    prediction = model.predict_log_proba(feature_vect)
    maxprob = max(prediction[0])
    if exp(maxprob) > classifier_threshold:
        for i in range(0, len(prediction[0])):
            if prediction[0][i] == maxprob:
                return {'class': labelencoder.inverse_transform(i), 'probability': exp(maxprob)}
    return None


'''
Aggregated categories:
politics
sports
business
world
entertainment/art
national/local
style/food/travel
science/technology/health
'''


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
                # 'ADV': 'r', 'RBR': 'r', 'RBS': 'r',
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
    ARTICLES_PER_CATEG0RY = 30000

    # #############################################################################
    # Load categories from the training set
    classes = ['world', 'politics', 'business', 'sports', 'entertainment/art',
               'national/local', 'style/food/travel', 'science/technology/health']

    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client["NewsAnalyzer"]

    # load the dataset
    articles = []
    for cat in classes:
        for aa in list(db.articles.aggregate([{"$match": {"category_aggregate": cat, "ground_truth": True}},
                                              {"$sample": {"size": ARTICLES_PER_CATEG0RY}}])):
            articles.append(aa)

    print('Training SVM on ' + str(len(articles)) + ' examples for ' + str(len(classes)) + ' classes...')

    # X -> features, y -> label
    X = []
    y = []

    # create features as concatenation of keywords and tags, labels as category
    for a in articles:
        if a['category_aggregate'] in classes:

            kw_concat = ' '.join(a['keywords'])
            if len(a['tags']) > 2:
                kw_concat = kw_concat + ' ' + ' '.join(a['tags'])

            # kw_concat = transform_keywords(kw_concat)

            X.append(kw_concat)
            y.append(a['category_aggregate'])

    # encode labels into integers
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    y_encoded = le.transform(y)

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1)

    # vectorize features
    count_vect = CountVectorizer(max_df=1.0, min_df=1, max_features=None, binary=True)
    X_train_counts = count_vect.fit_transform(X_train)
    # tfidf_transformer = TfidfTransformer(norm=None, use_idf=True)
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    print('Feature vector size: ' + str(len(count_vect.vocabulary_)))

    # training a linear SVM classifier
    from sklearn.svm import SVC
    svm_model_linear = SVC(kernel='linear', C=0.7, probability=True)
    svm_model_linear.fit(X_train_counts, y_train)
    svm_predictions = svm_model_linear.predict(count_vect.transform(X_test))

    # model accuracy for X_test
    accuracy = svm_model_linear.score(count_vect.transform(X_test), y_test)

    # creating a confusion matrix
    cm = confusion_matrix(y_test, svm_predictions)

    print('done')
    print('accuracy: ' + str(accuracy))

    # save model, vectors and label encoding to disk
    save_model(svm_model_linear, count_vect, le)


def main():
    create_model()


if __name__ == "__main__":
    timeStart = datetime.datetime.now()
    main()
    timeEnd = datetime.datetime.now()
    delta = timeEnd - timeStart
    print('Executed in ' + str(int(delta.total_seconds())) + 's')
