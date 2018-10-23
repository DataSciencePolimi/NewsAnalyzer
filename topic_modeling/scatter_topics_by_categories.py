import pickle
from ast import literal_eval
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pymongo
from gensim.models import LdaModel
from gensim import corpora

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["NewsAnalyzer"]

data_path = 'tmp/all_32k/'

with open(data_path+'pp_data_train.txt') as f:
    texts = [literal_eval(line) for line in f]

raw_texts = []
with open(data_path+'raw_data_train.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()
    for line in filecontents:
        # remove linebreak which is the last character of the string
        current_place = line[:-1]
        # add item to the list
        raw_texts.append(current_place)

targets = pickle.load(open(data_path+'target_train.sav', 'rb'))
encoder_translator = {0: 2, 1: 4, 2: 5, 3: 0, 4: 3, 5: 6, 6: 7, 7: 1}
target_conv = []
for t in targets:
    target_conv.append(encoder_translator[t])


dictionary = corpora.Dictionary.load(data_path + 'dictionary.dict')
lda_model = LdaModel.load(data_path + 'lda_models/100/LDA_model.lda')

topic_distribution = []
for d in texts:
    bow = dictionary.doc2bow(d)
    dst = np.zeros(100)
    for p in lda_model.get_document_topics(bow, minimum_probability=0.05):
        dst[p[0]] = p[1]
    topic_distribution.append(dst)

def scatter_3_classes():
    c0 = 3
    c1 = 4
    c2 = 7
    topic_0 = []
    topic_1 = []
    topic_2 = []
    for i in range(0, len(target_conv)):
        t = target_conv[i]
        if t == c0:
            topic_0.append(topic_distribution[i])
        elif t == c1:
            topic_1.append(topic_distribution[i])
        elif t == c2:
            topic_2.append(topic_distribution[i])

    topic_0 = topic_0[:200]
    topic_1 = topic_1[:200]
    topic_2 = topic_2[:200]

    concat = np.concatenate([topic_0, topic_1, topic_2])

    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(concat)
    c0_scaled = X_transformed[0:200, :]
    c1_scaled = X_transformed[200:400, :]
    c2_scaled = X_transformed[400:, :]

    fig, ax = plt.subplots()
    x = c0_scaled[:, 0]
    y = c0_scaled[:, 1]
    ax.scatter(x, y, label='technology')
    x = c1_scaled[:, 0]
    y = c1_scaled[:, 1]
    ax.scatter(x, y, label='entertainment')
    x = c2_scaled[:, 0]
    y = c2_scaled[:, 1]
    ax.scatter(x, y, label='style')

    plt.title('200 docs for category, 50-topics model (multi dimensional scaled)')
    ax.legend()
    plt.show()


def scatter_2_classes():
    c0 = 3
    c1 = 6
    topic_0 = []
    topic_1 = []
    for i in range(0, len(target_conv)):
        t = target_conv[i]
        if t == c0:
            topic_0.append(topic_distribution[i])
        elif t == c1:
            topic_1.append(topic_distribution[i])

    topic_0 = topic_0[:200]
    topic_1 = topic_1[:200]

    concat = np.concatenate([topic_0, topic_1])

    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(concat)
    c0_scaled = X_transformed[0:200, :]
    c1_scaled = X_transformed[200:400, :]

    fig, ax = plt.subplots()
    x = c0_scaled[:, 0]
    y = c0_scaled[:, 1]
    ax.scatter(x, y, label='technology')
    x = c1_scaled[:, 0]
    y = c1_scaled[:, 1]
    ax.scatter(x, y, label='sports')

    plt.title('200 docs for category, 50-topics model (multi dimensional scaled), min prob=0.01')
    ax.legend()
    plt.show()

def scatter_subclass():
    c = 3
    c_label = 'style/food/travel'
    subcategories = ['technology', 'science', 'health']
    topic_0 = []
    topic_1 = []
    topic_2 = []
    for i in range(0, len(target_conv)):
        t = target_conv[i]
        if t == c:
            rt = raw_texts[i]
            subclass = list(db.articles.aggregate([{'$match': {'ground_truth': True, 'category_aggregate': c_label}},
                                              {'$match': {'text': rt}}]))
            if len(subclass) > 0:
                if subclass[0]['category'] == subcategories[0]:
                    topic_0.append(topic_distribution[i])
                elif subclass[0]['category'] == subcategories[1]:
                    topic_1.append(topic_distribution[i])
                elif subclass[0]['category'] == subcategories[2]:
                    topic_2.append(topic_distribution[i])

    topic_0 = topic_0[:200]
    topic_1 = topic_1[:200]
    topic_2 = topic_2[:200]

    concat = np.concatenate([topic_0, topic_1, topic_2])

    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(concat)
    c0_scaled = X_transformed[0:200, :]
    c1_scaled = X_transformed[200:400, :]
    c2_scaled = X_transformed[400:, :]

    fig, ax = plt.subplots()
    x = c0_scaled[:, 0]
    y = c0_scaled[:, 1]
    ax.scatter(x, y, label=subcategories[0])
    x = c1_scaled[:, 0]
    y = c1_scaled[:, 1]
    ax.scatter(x, y, label=subcategories[1])
    x = c2_scaled[:, 0]
    y = c2_scaled[:, 1]
    ax.scatter(x, y, label=subcategories[2])

    plt.title('sub-categories, 50-topics model (multi dimensional scaled)')
    ax.legend()
    plt.show()


