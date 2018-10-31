from random import randint
import numpy as np
import pickle
import math
import scipy.stats
import scipy.spatial.distance as vd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from user_modeling import cluster_users as cluster
from user_modeling import evaluate_cluster_cohesion as eval_cluster

# CLASSIFIER PARAMETERS
feature_name = 'topics_class_source'

def normalize_to_sign(vector):
    vc = np.array(vector)
    v_min = vc.min()
    v_max = vc.max()
    result = []
    for i in range(0, len(vector)):
        result.append( 2 * ((vector[i] - v_min) / (v_max - v_min)) - 1)
    return result

def normalize(vector):
    vc = np.array(vector)
    v_min = vc.min()
    v_max = vc.max()
    result = []
    for i in range(0, len(vector)):
        result.append((vector[i] - v_min) / (v_max - v_min))
    return result


# all users
v_users = pickle.load(open('data/vectorized_users.sav', 'rb'))

# add composite features
for u in v_users:
    u['t_all'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300']))
    u['c_s'] = np.concatenate((u['class_vector'], u['source_vector']))
    u['t_rt_tl'] = np.concatenate((u['t_all'], [u['avg_RT']], [u['avg_text_length']]))
    u['t_rt_tl_pol'] = np.concatenate((u['t_rt_tl'], [u['avg_polarity']], [u['avg_subjectivity']]))
    u['t_c_s'] = np.concatenate((u['t_all'], u['c_s']))
    u['t_c_s_rt_tl'] = np.concatenate((u['t_rt_tl'], u['c_s']))
# get republican and democratic clusters
v_rep, v_dem = cluster.get_republican_democratic_clusters(v_users)


def evaluate_distance_difference():
    # compute centroids
    rep_centroid = eval_cluster.compute_centroid([u[feature_name] for u in v_rep])
    dem_centroid = eval_cluster.compute_centroid([u[feature_name] for u in v_dem])

    evaluated_users = []
    for u in v_users:
        # v_feature = smooth_zeros(u[feature_name])
        v_feature = u[feature_name]
        d_rep = eval_cluster.compute_distance(v_feature, rep_centroid, distance_metric='KL')
        d_dem = eval_cluster.compute_distance(v_feature, dem_centroid, distance_metric='KL')
        delta = d_rep - d_dem

        if delta and not math.isnan(delta) and not math.isinf(delta):
            evaluated_users.append((u['user_id'], delta))

    distances = [u[1] for u in evaluated_users]
    sns.distplot(normalize_to_sign(distances))

def evaluate_distance(centroid):
    evaluated_users = []
    for u in v_users:
        # v_feature = smooth_zeros(u[feature_name])
        v_feature = u[feature_name]
        distance = eval_cluster.compute_distance(v_feature, centroid, distance_metric='KL')

        if distance and not math.isnan(distance) and not math.isinf(distance):
            evaluated_users.append((u['user_id'], distance))

    distances = [u[1] for u in evaluated_users]
    sns.distplot(distances)


def train_test_classifier(republicans, democrats, feature_name):
    X = [(r[feature_name], 0) for r in republicans]
    X.extend([(d[feature_name], 1) for d in democrats])

    X = np.array(X)
    np.random.shuffle(X)

    y = [x[1] for x in X]
    X = [x[0] for x in X]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)

    from sklearn.linear_model import SGDClassifier
    clf_log_reg = SGDClassifier(n_jobs=-1, max_iter=5000, tol=0.0001, learning_rate='optimal', loss='modified_huber')
    clf_log_reg.fit(X_train, y_train)
    train_acc_log = clf_log_reg.score(X_train, y_train)
    test_acc_log = clf_log_reg.score(X_test, y_test)
    print('')
    print('LOGISTIC REGRESSION -------------------------------------')
    print('Feature: ' + feature_name)
    print('Train score: ' + str(train_acc_log))
    print('Test score: ' + str(test_acc_log))

    return clf_log_reg


def predict_random_user(cls, feature_name):
    user = v_users[randint(0, len(v_users))]
    return cls.predict_proba([user[feature_name]])


def predict(cls, feature_vector):
    return cls.predict_proba([feature_vector])


def evaluate_user_hashtags(user_id, db):
    rep_keywords = ['maga', 'kag', 'trump2020' 'americafirst', 'trumptrain', 'armyoftrump',
                    'votegop2018', 'backtheblue', 'redwaverising', 'trumpismypresident',
                    'trumpsupporters', 'conservativeandproud', 'makeamericared']
    dem_keywords = ['bluewave', 'theresistance', 'votebluetosaveamerica', 'notmypresident', 'voteblue',
                    'uniteblue', 'nevertrump', 'antitrump', 'liartrump', 'trumplies', 'trumpshutdown']
    rep_counter = 0
    dem_counter = 0
    for tw in db.tweet.find({'id_user': str(user_id)}):
        hashtags = [h['text'].lower() for h in tw['entities']['hashtags']]
        if any(x in hashtags for x in rep_keywords):
            rep_counter += 1
        if any(x in hashtags for x in dem_keywords):
            dem_counter += 1
    return {'rep_count': rep_counter, 'dem_count': dem_counter}


def evaluate_classifier_consistency(v_test_users, cls, feature_name, db):
    results = []
    counter = 0
    for t in v_test_users:
        hashtags = evaluate_user_hashtags(t['user_id'], db)
        prediction = predict(cls, t[feature_name])

        # rep by hashtags
        if hashtags['rep_count'] > hashtags['dem_count']:
            # predicted rep
            if prediction[0][0] > prediction[0][1]:
                results.append(1)
            else:
                results.append(0)
        # dem by hashtags
        if hashtags['rep_count'] < hashtags['dem_count']:
            # predicted dem
            if prediction[0][0] < prediction[0][1]:
                results.append(1)
            else:
                results.append(0)

        counter += 1
        if divmod(counter, 10)[1] == 0:
            print('-------------------------------')
            print('Analyzed users: ' + str(counter))
            print('Useful users found: ' + str(len(results)))
            print('current consistency: ' + str(sum(results) / max(1, len(results))))
    return sum(results) / len(results)


cls = train_test_classifier(v_rep, v_dem, feature_name='t_all')
# train_test_classifier(v_rep, v_dem, feature_name='t_rt_tl')
# train_test_classifier(v_rep, v_dem, feature_name='t_rt_tl_pol')
# train_test_classifier(v_rep, v_dem, feature_name='c_s')
# train_test_classifier(v_rep, v_dem, feature_name='t_c_s')
# train_test_classifier(v_rep, v_dem, feature_name='t_c_s_rt_tl')

pickle.dump(cls, open('models/rep_dem_t_all.sav', 'wb'))
# pickle.dump(cls_2, open('models/rep_dem_2.sav', 'wb'))

"""
cls = pickle.load(open('models/rep_dem_t_all.sav', 'rb'))
test_users = v_users[3000:6000]

import pymongo
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["NewsAnalyzer"]

evaluate_classifier_consistency(test_users, cls, 't_all', db)
"""
