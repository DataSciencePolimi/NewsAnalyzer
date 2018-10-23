import pymongo
import pickle
from datetime import datetime
from topic_modeling.preprocess_corpus import preprocess
from gensim.models import LdaModel
from gensim import corpora
import numpy as np
import pprint as pp

"""
    This script vectorize a user in the following spaces:
    1) Article classes space
    2) Sources space
    3) Topics space

    Requires:
    ----------
    twitter screen_name or user_id
    pre-trained bigram model
    pre-trained dictionary
    pre-trained LDA models
    
    Returns:
    ----------
    a structure of the vectorized user

"""

def print_delta_time():
    print(str(datetime.now() - timeStart))


def vectorize_user(screen_name=None, user_id=None, bigr_mod=None, db=None,
                   dictionary=None, t_mod50=None, t_mod100=None, t_mod200=None, t_mod300=None):

    try:

        if not user_id and not screen_name:
            print('You need to provide a user_id or screen_name')
            return None

        if screen_name and not user_id:
            mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
            db = mongo_client["NewsAnalyzer"]
            user_id = db.user.find_one({'screen_name': screen_name})
            if not user_id:
                print('User not found')
                return None

        # list of user tweets
        user_tweets = list(db.tweet.find({"id_user": str(user_id)}))

        # generate pair tweet article
        tweet_article = []
        article_texts = []
        RT_size = 0
        tw_text_size = 0
        tw_polarity = 0
        tw_subjectivity = 0
        for t in user_tweets:
            try:
                if t['RT']:
                    RT_size += 1
                else:
                    tw_text_size += len(t['text'])
                    tw_polarity += t['sentiment_polarity']
                    tw_subjectivity += t['sentiment_subjectivity']

                article = db.articles.find_one({"_id": t["article"]})
                if article:
                    tweet_article.append((t, article))
                    article_texts.append(article['text'])
            except:
                pass

        if len(user_tweets) < 1 or len(tweet_article) < 1:
            return None

        # get mean text lenght
        try:
            tw_text_size = tw_text_size / (len(user_tweets)-RT_size)
            tw_polarity = tw_polarity / (len(user_tweets)-RT_size)
            tw_subjectivity = tw_subjectivity / (len(user_tweets) - RT_size)

        except ZeroDivisionError:
            tw_text_size = 0
        # get percentage of RT
        RT_size = RT_size / len(user_tweets)

        # preprocess articles
        pp_texts, bigram_model = preprocess(sentences=article_texts, bigram_model=bigr_mod)

        # initialize vectors
        class_vector = np.zeros(8)
        src_vector = np.zeros(69)
        class_counter = 0
        src_counter = 0
        topic_matrix_50 = []
        topic_matrix_100 = []
        topic_matrix_200 = []
        topic_matrix_300 = []

        for i in range(0, len(tweet_article)):
            try:
                t = tweet_article[i]
                # count classes
                cls = t[1]['category_aggregate']
                if cls:
                    cls_index = db.category_aggregate.find_one({'name': cls})['_id']
                    class_vector[cls_index] += 1
                    class_counter += 1

                # count sources
                src = t[1]['source_name']
                if src:
                    src_index = db.sources.find_one({'name': src})['source_index']
                    src_vector[src_index] += 1
                    src_counter += 1

                # evaluate topics
                d = dictionary.doc2bow(pp_texts[i])
                dst_50 = np.zeros(50)
                dst_100 = np.zeros(100)
                dst_200 = np.zeros(200)
                dst_300 = np.zeros(300)
                for p in t_mod50.get_document_topics(d, minimum_probability=0.05):
                    dst_50[p[0]] = p[1]
                topic_matrix_50.append(dst_50)
                for p in t_mod100.get_document_topics(d, minimum_probability=0.05):
                    dst_100[p[0]] = p[1]
                topic_matrix_100.append(dst_100)
                for p in t_mod200.get_document_topics(d, minimum_probability=0.05):
                    dst_200[p[0]] = p[1]
                topic_matrix_200.append(dst_200)
                for p in t_mod300.get_document_topics(d, minimum_probability=0.05):
                    dst_300[p[0]] = p[1]
                topic_matrix_300.append(dst_300)
            except:
                pass

        n_doc = len(pp_texts)
        # extract centroids
        centroid_50 = np.true_divide(np.sum(topic_matrix_50, axis=0), n_doc)
        centroid_100 = np.true_divide(np.sum(topic_matrix_100, axis=0), n_doc)
        centroid_200 = np.true_divide(np.sum(topic_matrix_200, axis=0), n_doc)
        centroid_300 = np.true_divide(np.sum(topic_matrix_300, axis=0), n_doc)

        # normalize classes and sources
        if class_counter > 0:
            class_vector = [x / class_counter for x in class_vector]
        if src_counter > 0:
            src_vector = [x / src_counter for x in src_vector]

    except:
        return None

    return {'user_id': user_id, 'avg_RT': RT_size, 'avg_text_length': tw_text_size,
            'avg_polarity': tw_polarity, 'avg_subjectivity': tw_subjectivity,
            'source_vector': src_vector, 'class_vector': class_vector,
            'topics_50': centroid_50,
            'topics_100': centroid_100,
            'topics_200': centroid_200,
            'topics_300': centroid_300
            }


def main():
    v_user_list = []
    counter = 0

    data_path = '../topic_modeling/tmp/all_32k/'

    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client["NewsAnalyzer"]

    # load models
    bigram_model = pickle.load(open(data_path + 'bigram_model.sav', 'rb'))
    dictionary = corpora.Dictionary.load(data_path + 'dictionary.dict')
    lda_model_50 = LdaModel.load(data_path + 'lda_models/50/LDA_model.lda')
    lda_model_100 = LdaModel.load(data_path + 'lda_models/100/LDA_model.lda')
    lda_model_200 = LdaModel.load(data_path + 'lda_models/200/LDA_model.lda')
    lda_model_300 = LdaModel.load(data_path + 'lda_models/300/LDA_model.lda')

    for user in db.user.aggregate([{"$sample": {"size": 200000}}]):
        v_user = vectorize_user(user_id=user['_id'], db=db, bigr_mod=bigram_model, dictionary=dictionary,
                                t_mod50=lda_model_50, t_mod100=lda_model_100, t_mod200=lda_model_200, t_mod300=lda_model_300)
        if v_user:
            v_user_list.append(v_user)
        counter += 1
        if divmod(counter, 10)[1] == 0:
            print('Done users: ' + str(counter))
        if divmod(counter, 2000)[1] == 0:
            pickle.dump(v_user_list, open('vectorized_users.sav', 'wb'))

    pickle.dump(v_user_list, open('vectorized_users.sav', 'wb'))


if __name__ == "__main__":
    timeStart = datetime.now()
    main()
    timeEnd = datetime.now()
    delta = timeEnd - timeStart
    print('Executed in ' + str(int(delta.total_seconds())) + 's')

