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
        r = randint(0, len(accounts)-1)
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

def compute_distance(v1, v2, distance_metric='euclidean'):
    if distance_metric == 'manhattan':
        distance = vd.cityblock(v1, v2)
    elif distance_metric == 'cosine':
        distance = vd.cosine(v1, v2)
    elif distance_metric == 'KL':
        distance = scipy.stats.entropy(v1, v2)
    else:
        distance = vd.euclidean(v1, v2)
    return distance


def rank_by_distance(tuples, centroid, distance_metric='euclidean'):
    result = []
    reverse = False
    for t in tuples:
        distance = compute_distance(t[1], centroid, distance_metric=distance_metric)
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


def plot_precision_recall(data_list, title=None, labels=('democrats', 'republicans')):
    for data in data_list:
        plt.plot([d[1] for d in data], [d[0] for d in data], linewidth=2.0)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if title:
        plt.title(title)
    plt.legend(labels=labels)
    plt.show()


def evaluate_feature(mixed_cluster=None, pure_cluster=None, feature_name=None, distance_type='KL', cluster_size=50):
    if not mixed_cluster or not pure_cluster or not feature_name:
        return None

    # compute centroid
    centroid = compute_centroid([u[feature_name] for u in pure_cluster])

    cluster_rank = rank_by_distance([(u['user_id'], u[feature_name]) for u in mixed_cluster], centroid,
                                distance_metric=distance_type)
    positive_ids = [u['user_id'] for u in pure_cluster]
    cluster_eval = []
    for t in cluster_rank:
        if t[0] in positive_ids:
            cluster_eval.append(1)
        else:
            cluster_eval.append(0)

    print(cluster_eval)

    cluster_prec_rec = []
    recall = 0
    threshold = 5
    while recall < 1:
        p, r = get_precision_recall(cluster_eval, threshold, true_size=cluster_size)
        cluster_prec_rec.append((p, r))
        recall = r
        threshold += 5

    return cluster_prec_rec


def compute_mean_distance(user, user_group, feature=None):
    distance = 0
    for ug in user_group:
        distance += vd.cosine(user[feature], ug[feature])
    return distance / len(user_group)


def main():
    cluster_size = 50
    random_sample_size = 250
    distance_type = 'KL'


    # all users
    v_users = pickle.load(open('data/vectorized_users.sav', 'rb'))

    # get republican and democratic clusters
    v_rep, v_dem = cluster.get_republican_democratic_clusters(v_users)

    # add composite features
    for u in v_rep:
        u['topics_all'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300']))
        u['topics_100_200'] = np.concatenate((u['topics_100'], u['topics_200']))
        u['topics_class_source'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300'], u['class_vector'], u['source_vector']))
    for u in v_dem:
        u['topics_all'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300']))
        u['topics_100_200'] = np.concatenate((u['topics_100'], u['topics_200']))
        u['topics_class_source'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300'], u['class_vector'], u['source_vector']))


    # slice data
    v_rep = v_rep[:cluster_size]
    v_dem_all = [ut for ut in v_dem]
    v_dem = v_dem[:cluster_size]

    # add random samples
    rep_sample, rep_random = sample_random_accounts(v_rep, v_users, sample_size=random_sample_size)
    dem_sample, dem_random = sample_random_accounts(v_dem, v_users, sample_size=random_sample_size)
    rep_dem_sample, rep_dem_random = sample_random_accounts(v_rep, v_dem_all, sample_size=random_sample_size)


    # add composite features
    for u in rep_sample:
        u['topics_all'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300']))
        u['topics_100_200'] = np.concatenate((u['topics_100'], u['topics_200']))
        u['topics_class_source'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300'], u['class_vector'], u['source_vector']))
    for u in dem_sample:
        u['topics_all'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300']))
        u['topics_100_200'] = np.concatenate((u['topics_100'], u['topics_200']))
        u['topics_class_source'] = np.concatenate((u['topics_50'], u['topics_100'], u['topics_200'], u['topics_300'], u['class_vector'], u['source_vector']))

    """
    # republicans and democratics VS random users
    result_topic_50 = [evaluate_feature(rep_sample, v_rep, 'topics_50'), evaluate_feature(dem_sample, v_dem, 'topics_50')]
    result_topic_100 = [evaluate_feature(rep_sample, v_rep, 'topics_100'), evaluate_feature(dem_sample, v_dem, 'topics_100')]
    result_topic_200 = [evaluate_feature(rep_sample, v_rep, 'topics_200'), evaluate_feature(dem_sample, v_dem, 'topics_200')]
    result_topic_300 = [evaluate_feature(rep_sample, v_rep, 'topics_300'), evaluate_feature(dem_sample, v_dem, 'topics_300')]
    result_topic_all = [evaluate_feature(rep_sample, v_rep, 'topics_all'), evaluate_feature(dem_sample, v_dem, 'topics_all')]
    result_feature_all = [evaluate_feature(rep_sample, v_rep, 'topics_class_source'), evaluate_feature(dem_sample, v_dem, 'topics_class_source')]
    
    plot_precision_recall(result_topic_50, title='feature: 50 topics')
    plot_precision_recall(result_topic_100, title='feature: 100 topics')
    plot_precision_recall(result_topic_200, title='feature: 200 topics')
    plot_precision_recall(result_topic_300, title='feature: 300 topics')
    plot_precision_recall(result_topic_all, title='feature: All topics')
    plot_precision_recall(result_feature_all, title='feature: topics, classes, sources')
    """

    # republican VS democratics
    result_topic_50 = evaluate_feature(rep_dem_sample, v_rep, 'topics_50')
    result_topic_100 = evaluate_feature(rep_dem_sample, v_rep, 'topics_100')
    result_topic_200 = evaluate_feature(rep_dem_sample, v_rep, 'topics_200')
    result_topic_300 = evaluate_feature(rep_dem_sample, v_rep, 'topics_300')
    result_topic_100_200 = evaluate_feature(rep_dem_sample, v_rep, 'topics_100_200')
    result_topic_all = evaluate_feature(rep_dem_sample, v_rep, 'topics_all')
    result_feature_all = evaluate_feature(rep_dem_sample, v_rep, 'topics_class_source')

    plot_precision_recall([result_topic_50], title='feature: 50 topics', labels=['republicans'])
    plot_precision_recall([result_topic_100], title='feature: 100 topics', labels=['republicans'])
    plot_precision_recall([result_topic_200], title='feature: 200 topics', labels=['republicans'])
    plot_precision_recall([result_topic_300], title='feature: 300 topics', labels=['republicans'])
    plot_precision_recall([result_topic_100_200], title='feature: 100+200 topics', labels=['republicans'])
    plot_precision_recall([result_topic_all], title='feature: All topics', labels=['republicans'])
    plot_precision_recall([result_feature_all], title='feature: topics, classes, sources', labels=['republicans'])


if __name__ == "__main__":
    main()
