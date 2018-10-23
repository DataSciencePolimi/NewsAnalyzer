import numpy as np
import pymongo
import pickle
import pprint as pp


mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["NewsAnalyzer"]

def evaluate_domain_expertise(user_vector):
    '''
    This function evaluates expertise of a user in all the topics

    :param user_vector: vectorized user
    :return: evaluated expertise for each topic
    '''

    epsilon = 0.00001
    n_tweets = db.tweet.find({'id_user': str(user_vector['user_id'])}).count()

    class_coeff = max(user_vector['class_vector'])
    source_coeff = (1 - max(user_vector['source_vector']) + epsilon)
    n_tw_coeff = min(n_tweets, 300) / 300
    expertise_coeff = ( class_coeff + source_coeff + n_tw_coeff ) + ((1 - user_vector['avg_RT'] + epsilon) + ((user_vector['avg_text_length'] / 240) + epsilon))

    print('class coeff: ' + str(class_coeff))
    print('source coeff: ' + str(source_coeff))
    print('n_tweet coeff: ' + str(n_tw_coeff))
    print('RT: ' + str(user_vector['avg_RT']))
    print('text len: ' + str(user_vector['avg_text_length']))
    print('EXP: ' + str(expertise_coeff))
    print('--------------')

    return expertise_coeff


def evaluate_publication_posting_delta(user_vector):
    delta = []
    user_id = int(user_vector['user_id'])
    for tw in db.tweet.find({'id_user': str(user_id)}):
        tw_date = 0
        art = db.articles.find({'id': tw['article']})
        art_date = 0
        delta.append(art_date - tw_date)
    return np.true_divide(np.sum(delta), len(delta))


#def get_temporal_tweet_distribution(user_vector):



#def get_temporal_topics_distribution(user_vector):



#def get_temporal_category_distribution(user_vector):



v_users = pickle.load(open('data/vectorized_users.sav', 'rb'))
for i in range(0, 10):
    evaluate_domain_expertise(v_users[i])

pp.pprint(v_users[0])
pp.pprint(v_users[7])


