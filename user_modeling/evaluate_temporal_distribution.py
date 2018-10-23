from random import randint
import numpy as np
import pymongo
import pickle
from datetime import datetime
import scipy.stats
import scipy.spatial.distance as vd
import matplotlib.pyplot as plt
from user_modeling import cluster_users as cluster
from topic_modeling.preprocess_corpus import preprocess
from gensim.models import LdaModel
from gensim import corpora


def get_temporal_distribution(categories, user_ids=None, lda_model=None, bigram_model=None, dictionary=None, save_name=None):

    cat_distr = {}
    topics_distr = []
    tweet_count = []

    for c in categories:
        cat_distr[c] = []

    # month
    for m in range(8, 10):
        # day
        for d in range(1, 31):
            # tweets by day
            if d < 10:
                d_str = "0" + str(d)
            else:
                d_str = str(d)

            print('day: ' + str(m) + ' / ' + d_str)

            for c in categories:
                cat_distr[c].append(0)

            articles_texts = []

            if user_ids:
                tw_by_day = db.tweet.find({'id_user': {'$in': user_ids},  "create_at": {"$regex": ".*2018-0"+str(m)+"-"+d_str+".*"}})
            else:
                tw_by_day = db.tweet.find({"create_at": {"$regex": ".*2018-0" + str(m) + "-" + d_str + ".*"}})

            tweet_count.append(tw_by_day.count())

            for tw in tw_by_day:
                try:
                    art = db.articles.find_one({'_id': tw['article']})
                    if art and 'category_aggregate' in art and art['category_aggregate']:
                        cat_distr[art['category_aggregate']][-1] += 1
                        articles_texts.append(art['text'])
                except KeyError:
                    continue

            topics_vector = np.zeros(100)

            if len(articles_texts) > 0:
                pp_texts, bigram_model = preprocess(sentences=articles_texts, bigram_model=bigram_model)
            else:
                pp_texts = []

            for i in range(0, len(pp_texts)):
                doc2bow = dictionary.doc2bow(pp_texts[i])
                for p in lda_model.get_document_topics(doc2bow, minimum_probability=0.05):
                    topics_vector[p[0]] += p[1]

            # mean of topics of the day
            topics_vector = np.true_divide(topics_vector, len(pp_texts))

            # append vector of the day
            topics_distr.append(topics_vector)

    result = {
        'tweet_count': tweet_count,
        'categories': cat_distr,
        'topics': topics_distr
    }
    if save_name:
        pickle.dump(result, open('data/'+save_name+'.sav', 'wb'))

    return result


def analyze_democratic_republican():
    # all users
    v_users = pickle.load(open('data/vectorized_users.sav', 'rb'))

    # get republican and democratic clusters
    v_rep, v_dem = cluster.get_republican_democratic_clusters(v_users)

    rep_ids = [str(u['user_id']) for u in v_rep]
    dem_ids = [str(u['user_id']) for u in v_dem]

    get_temporal_distribution(categories=categories, lda_model=lda_model, dictionary=dictionary,
                              bigram_model=bigram_model, user_ids=rep_ids, save_name='rep_tmp_dst')
    get_temporal_distribution(categories=categories, lda_model=lda_model, dictionary=dictionary,
                              bigram_model=bigram_model, user_ids=dem_ids, save_name='dem_tmp_dst')


def main():
    # get_temporal_distribution(categories=categories, lda_model=lda_model, dictionary=dictionary, bigram_model=bigram_model)
    analyze_democratic_republican()


if __name__ == "__main__":
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client["NewsAnalyzer"]

    data_path = '../topic_modeling/tmp/all_32k/'

    categories = [c['name'] for c in list(db.category_aggregate.find())]

    # load models
    bigram_model = pickle.load(open(data_path + 'bigram_model.sav', 'rb'))
    dictionary = corpora.Dictionary.load(data_path + 'dictionary.dict')
    lda_model = LdaModel.load(data_path + 'lda_models/100/LDA_model.lda')

    timeStart = datetime.now()
    main()
    timeEnd = datetime.now()
    delta = timeEnd - timeStart
    print('Executed in ' + str(int(delta.total_seconds())) + 's')


