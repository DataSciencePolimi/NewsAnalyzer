import pymongo
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def create(art_per_category=1000, classes=None, test_size=0.05, save_path='tmp/'):

    EXCLUDE_SOURCES = ['The Guardian',
                       'CBC News',
                       'Reuters',
                       'BBC',
                       'The Globe and Mail',
                       'New York Daily News',
                       'Los Angeles Times',
                       'News Channel 8'
                       ]

    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client["NewsAnalyzer"]

    # load the dataset
    articles = []
    target_categories = []
    for cat in classes:
        for aa in list(db.articles.aggregate([{"$match":
                                               {"category_aggregate": cat,
                                                "ground_truth": True, "source_name": { "$nin": EXCLUDE_SOURCES}
                                                }},
                                              {"$sample": {"size": art_per_category}}])):
            articles.append(aa['text'].replace('\n', ' ').replace('\r', ''))
            target_categories.append(aa['category_aggregate'])

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(articles, target_categories, test_size=test_size)

    # pickle.dump(articles, open(file_name, 'wb'))
    with open(save_path+'raw_data_train.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % a for a in X_train)

    with open(save_path+'raw_data_test.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % a for a in X_test)

    # encode labels into integers
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    y_en_train = le.transform(y_train)
    y_en_test = le.transform(y_test)
    pickle.dump(le, open(save_path+'labelencoder.sav', 'wb'))
    pickle.dump(y_en_train, open(save_path+'target_train.sav', 'wb'))
    pickle.dump(y_en_test, open(save_path+'target_test.sav', 'wb'))

    print('Sampled raw data, total count: ' + str(len(articles)))


