from random import randint
import numpy as np
import pickle
import scipy.stats
import scipy.spatial.distance as vd
import matplotlib.pyplot as plt
from user_modeling import cluster_users as cluster

"""
    This script evaluates cohesion of a cluster of users

"""


def sample_random_accounts(base_cluster, accounts, sample_size=700):
    cluster_ids = [u['user_id'] for u in base_cluster]
    random_users = []
    i = 0
    while i < sample_size:
        r = randint(0, len(accounts))
        if accounts[r]['user_id'] not in cluster_ids:
            random_users.append(accounts[r])
            i += 1

    mixed_users = random_users
    mixed_users.extend(base_cluster)
    np.random.shuffle(mixed_users)

    return mixed_users, random_users

def compute_centroid(vector):
    return np.true_divide(np.sum(vector, axis=0), len(vector))


def compute_dispersion_index(cluster_centroid, random_centroid):
    return random_centroid / (cluster_centroid + random_centroid)


def rank_by_distance(tuples, centroid, distance_metric='euclidean'):
    result = []
    reverse = False
    for t in tuples:
        if distance_metric == 'manhattan':
            distance = vd.cityblock(t[1], centroid)
        elif distance_metric == 'cosine':
            distance = vd.cosine(t[1], centroid)
        elif distance_metric == 'KL':
            distance = scipy.stats.entropy(t[1], centroid)
        else:
            distance = vd.euclidean(t[1], centroid)
        result.append((t[0], distance))

    return sorted(result, key=lambda tup: tup[1], reverse=reverse)


def get_precision_recall(eval_vector, threshold, true_size=300):
    tp = 0
    if threshold == 0:
        return 0, 0
    for i in range(0, threshold):
        if eval_vector[i] == 1:
            tp += 1
    return tp/threshold, tp/true_size


def plot_precision_recall(data_list, title=None):
    for data in data_list:
        plt.plot([d[1] for d in data], [d[0] for d in data])
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if title:
        plt.title(title)
    plt.show()


def evaluate_feature(feature_name):
    # compute centroids
    rep_centroid = compute_centroid([u[feature_name] for u in v_rep])
    dem_centroid = compute_centroid([u[feature_name] for u in v_dem])

    # compute dispersion indexes
    '''
    rep_internal_mean_distance = np.true_divide( np.sum([compute_mean_distance(r, v_rep, feature_name) for r in v_rep]), len(v_rep))
    rep_external_mean_distance = np.true_divide(np.sum([compute_mean_distance(r, rep_random, feature_name) for r in rep_random]), len(v_rep))
    rep_dispersion = compute_dispersion_index(rep_internal_mean_distance, rep_external_mean_distance)
    dem_internal_mean_distance = np.true_divide(np.sum([compute_mean_distance(r, v_dem, feature_name) for r in v_dem]), len(v_dem))
    dem_external_mean_distance = np.true_divide(np.sum([compute_mean_distance(r, dem_random, feature_name) for r in dem_random]), len(v_dem))
    dem_dispersion = compute_dispersion_index(dem_internal_mean_distance, dem_external_mean_distance)

    print('Feature: ' + feature_name + ' REP dispersion: ' + str(rep_dispersion))
    print('Feature: ' + feature_name + ' DEM dispersion: ' + str(dem_dispersion))
    '''

    rep_rank = rank_by_distance([(u['user_id'], u[feature_name]) for u in rep_sample], rep_centroid,
                                distance_metric=distance_type)
    rep_ids = [u['user_id'] for u in v_rep]
    rep_eval = []
    for t in rep_rank:
        if t[0] in rep_ids:
            rep_eval.append(1)
        else:
            rep_eval.append(0)

    dem_rank = rank_by_distance([(u['user_id'], u[feature_name]) for u in dem_sample], dem_centroid,
                                distance_metric=distance_type)
    dem_ids = [u['user_id'] for u in v_dem]
    dem_eval = []
    for t in dem_rank:
        if t[0] in dem_ids:
            dem_eval.append(1)
        else:
            dem_eval.append(0)

    print(rep_eval)
    print(dem_eval)

    rep_prec_rec = []
    recall = 0
    threshold = 5
    while recall < 1:
        p, r = get_precision_recall(rep_eval, threshold, true_size=cluster_size)
        rep_prec_rec.append((p, r))
        recall = r
        threshold += 5

    dem_prec_rec = []
    recall = 0
    threshold = 5
    while recall < 1:
        p, r = get_precision_recall(dem_eval, threshold, true_size=cluster_size)
        dem_prec_rec.append((p, r))
        recall = r
        threshold += 5

    return [rep_prec_rec, dem_prec_rec]


def compute_mean_distance(user, user_group, feature=None):
    distance = 0
    for ug in user_group:
        distance += vd.cosine(user[feature], ug[feature])
    return distance / len(user_group)


cluster_size = 50
random_sample_size = 300
distance_type = 'KL'


# all users
v_users = pickle.load(open('data/vectorized_users.sav', 'rb'))

# get republican and democratic clusters
v_rep, v_dem = cluster.get_republican_democratic_clusters(v_users)

# slice data
v_rep = v_rep[:cluster_size]
v_dem = v_dem[:cluster_size]

# add composite features
for u in v_rep:
    u['topics_all'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300']))
    u['topics_class_source'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300'], u['class_vector'], u['source_vector']))
for u in v_dem:
    u['topics_all'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300']))
    u['topics_class_source'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300'], u['class_vector'], u['source_vector']))

# add random samples
rep_sample, rep_random = sample_random_accounts(v_rep, v_users, sample_size=random_sample_size)
dem_sample, dem_random = sample_random_accounts(v_dem, v_users, sample_size=random_sample_size)

# add composite features
for u in rep_sample:
    u['topics_all'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300']))
    u['topics_class_source'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300'], u['class_vector'], u['source_vector']))
for u in dem_sample:
    u['topics_all'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300']))
    u['topics_class_source'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300'], u['class_vector'], u['source_vector']))


result_topic_50 = evaluate_feature('topics_50')
result_topic_100 = evaluate_feature('topics_100')
result_topic_200 = evaluate_feature('topics_200')
result_topic_300 = evaluate_feature('topics_300')
result_topic_all = evaluate_feature('topics_all')
result_feature_all = evaluate_feature('topics_class_source')

plot_precision_recall(result_topic_50, title='feature: 50 topics')
plot_precision_recall(result_topic_100, title='feature: 100 topics')
plot_precision_recall(result_topic_200, title='feature: 200 topics')
plot_precision_recall(result_topic_300, title='feature: 300 topics')
plot_precision_recall(result_topic_all, title='feature: All topics')
plot_precision_recall(result_feature_all, title='feature: topics, classes, sources')

