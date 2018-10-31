import pickle
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pymongo

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["NewsAnalyzer"]

'''
users = pickle.load(open('vectorized_users.sav', 'rb'))
categories = db.category_aggregate.find()

id_stack = np.stack([v['user_id'] for v in users])
class_stack = np.stack([v['class_vector'] for v in users])
src_stack = np.stack([v['source_vector'] for v in users])
topic50_stack = np.stack([v['topics_50'] for v in users])
topic100_stack = np.stack([v['topics_100'] for v in users])
topic200_stack = np.stack([v['topics_200'] for v in users])
sentiment_stack = np.stack([v['avg_polarity'] for v in users])
'''

def users_by_category(category, user_list, class_threshold=0.5):
    category_id = db.category_aggregate.find_one({'name': category})['_id']

    id_vector = []
    topic_vector50 = []
    topic_vector100 = []
    topic_vector200 = []
    for i in range(0, len(user_list)):
        if user_list[i]['class_vector'][category_id] > class_threshold:
            id_vector.append(user_list[i]['user_id'])
            topic_vector50.append(user_list[i]['topics_50'])
            topic_vector100.append(user_list[i]['topics_100'])
            topic_vector200.append(user_list[i]['topics_200'])
    return id_vector, topic_vector50, topic_vector100, topic_vector200

def scatter_by_sex(ids, topic_stack):
    gender1_topics = []
    gender2_topics = []

    for i in range(0, len(ids)):
        user_info = db.user.find_one({'_id': int(ids[i])})
        if user_info and 't_gender' in user_info:
            sex = user_info['t_gender']
            if sex == 1:
                gender1_topics.append(topic_stack[i])
            else:
                gender2_topics.append(topic_stack[i])

    concat = np.concatenate([gender1_topics, gender2_topics])
    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(concat)
    c0_scaled = X_transformed[0:len(gender1_topics), :]
    c1_scaled = X_transformed[len(gender1_topics):, :]

    plt.scatter(c0_scaled[:, 0], c0_scaled[:, 1], label='female', c='deeppink')
    plt.scatter(c1_scaled[:, 0], c1_scaled[:, 1], label='male', c='royalblue')
    plt.legend(loc='best')
    plt.show()

def scatter_by_age(ids, topic_stack):
    age1_topics = []
    age2_topics = []
    age3_topics = []

    for i in range(0, len(ids)):
        user_info = db.user.find_one({'_id': int(ids[i])})
        if user_info and 't_age' in user_info:
            age = user_info['t_age']
            if age < 30:
                age1_topics.append(topic_stack[i])
            elif (age > 29) and (age < 50):
                age2_topics.append(topic_stack[i])
            else:
                age3_topics.append(topic_stack[i])

    concat = np.concatenate([age1_topics, age2_topics, age3_topics])
    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(concat)
    c0_scaled = X_transformed[0:len(age1_topics), :]
    c1_scaled = X_transformed[len(age1_topics):len(age1_topics)+len(age2_topics), :]
    c2_scaled = X_transformed[len(age1_topics) + len(age2_topics):, :]

    plt.scatter(c0_scaled[:, 0], c0_scaled[:, 1], label='<30')
    plt.scatter(c1_scaled[:, 0], c1_scaled[:, 1], label='>30, <50')
    plt.scatter(c2_scaled[:, 0], c2_scaled[:, 1], label='>50')
    plt.legend(loc='best')
    plt.show()

def plot_categories_by_sex(ids, category_stack):
    gender1_cat = []
    gender2_cat = []

    for i in range(0, len(ids)):
        user_info = db.user.find_one({'_id': int(ids[i])})
        if user_info and 't_gender' in user_info:
            sex = user_info['t_gender']
            if sex == 1:
                gender1_cat.append(category_stack[i])
            else:
                gender2_cat.append(category_stack[i])

    n_gender1 = len(gender1_cat)
    n_gender2 = len(gender2_cat)

    mean_gender1 = np.true_divide(np.sum(gender1_cat, axis=0), n_gender1)
    mean_gender2 = np.true_divide(np.sum(gender2_cat, axis=0), n_gender2)

    N = 8
    fig, ax = plt.subplots()

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars
    p1 = ax.bar(ind, mean_gender2, width, color='royalblue')

    p2 = ax.bar(ind + width, mean_gender1, width, color='deeppink')

    ax.set_title('Means by category and gender')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('politics', 'world', 'business', 's/t/h', 'e/a', 'n/l', 'sports', 's/f/t'))

    ax.legend((p1[0], p2[0]), ('Men', 'Women'))
    ax.autoscale_view()

    plt.show()

def plot_categories_by_age(ids, category_stack):
    age1_cat = []
    age2_cat = []
    age3_cat = []

    for i in range(0, len(ids)):
        user_info = db.user.find_one({'_id': int(ids[i])})
        if user_info and 't_age' in user_info:
            age = user_info['t_age']
            if age < 30:
                age1_cat.append(category_stack[i])
            elif (age > 29) and (age < 50):
                age2_cat.append(category_stack[i])
            else:
                age3_cat.append(category_stack[i])

    n1 = len(age1_cat)
    n2 = len(age2_cat)
    n3 = len(age3_cat)

    mean_1 = np.true_divide(np.sum(age1_cat, axis=0), n1)
    mean_2 = np.true_divide(np.sum(age2_cat, axis=0), n2)
    mean_3 = np.true_divide(np.sum(age3_cat, axis=0), n3)

    N = 8
    fig, ax = plt.subplots()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25  # the width of the bars
    p1 = ax.bar(ind, mean_1, width)
    p2 = ax.bar(ind + width, mean_2, width)
    p3 = ax.bar(ind + width*2, mean_3, width)

    ax.set_title('Means by category and age')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('politics', 'world', 'business', 's/t/h', 'e/a', 'n/l', 'sports', 's/f/t'))

    ax.legend((p1[0], p2[0], p3[0]), ('<30', '>30, <50', '>50'))
    ax.autoscale_view()

    plt.show()

def plot_sentiment(polarity_list, labels=None):
    means = []
    bars = []
    N = len(polarity_list)
    fig, ax = plt.subplots()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.20  # the width of the bars
    for i in range(0, len(polarity_list)):
        means.append(np.true_divide(np.sum(polarity_list[i], axis=0), len(polarity_list[i])))
        p = ax.bar(ind, means[i], width)
        bars.append(p)

    ax.set_title('Mean polarity')
    ax.set_xticklabels('tweet polarity')
    if labels:
        ax.legend((p[0] for p in bars), (l for l in labels))
    ax.autoscale_view()

    plt.show()


def plot_sentiment_by_sex(ids, polarity):
    gender1_cat = []
    gender2_cat = []

    for i in range(0, len(ids)):
        user_info = db.user.find_one({'_id': int(ids[i])})
        if user_info and 't_gender' in user_info:
            sex = user_info['t_gender']
            if sex == 1:
                gender1_cat.append(polarity[i])
            else:
                gender2_cat.append(polarity[i])

    n_gender1 = len(gender1_cat)
    n_gender2 = len(gender2_cat)

    mean_gender1 = np.true_divide(np.sum(gender1_cat, axis=0), n_gender1)
    mean_gender2 = np.true_divide(np.sum(gender2_cat, axis=0), n_gender2)

    N = 1
    fig, ax = plt.subplots()

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars
    p1 = ax.bar(ind, mean_gender2, width, color='royalblue')

    p2 = ax.bar(ind + width, mean_gender1, width, color='deeppink')

    ax.set_title('Mean polarity by gender')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels('tweet polarity')

    ax.legend((p1[0], p2[0]), ('Men', 'Women'))
    ax.autoscale_view()

    plt.show()

def plot_bot_by_categories(ids, category_stack, min_bot_prob):
    bot_category_stack = []
    for i in range(0, len(ids)):
        usr = db.user.find_one({"bot_score": {"$exists": True}, '_id': int(ids[i])})
        if usr and usr['bot_score']['english'] > min_bot_prob:
            bot_category_stack.append(category_stack[i])

    print('Found ' + str(len(bot_category_stack)) + ' probable bots')
    mean_categories_bot = np.true_divide(np.sum(bot_category_stack, axis=0), len(bot_category_stack))
    mean_general = np.true_divide(np.sum(category_stack, axis=0), len(category_stack))
    N = 8
    fig, ax = plt.subplots()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25  # the width of the bars
    p1 = ax.bar(ind, mean_categories_bot, width)
    p2 = ax.bar(ind + width, mean_general, width)

    ax.set_title('Means categories for bots')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('politics', 'world', 'business', 's/t/h', 'e/a', 'n/l', 'sports', 's/f/t'))

    ax.legend((p1[0], p2[0]), ('bots', 'any'))
    ax.autoscale_view()

    plt.show()


def get_users_by_topic(ids, topic_stack, topid=0, topic_threshold=0.3):
    users_by_topic = []
    for i in range(0, len(ids)):
        # if topic_stack[i][topid] > topic_threshold:
        if topic_stack[i][topid] == max(topic_stack[i]):
            users_by_topic.append(ids[i])
    return users_by_topic


def plot_distribution(data):
    data = np.sort(data)
    unique, counts = np.unique(data, return_counts=True)
    plt.fill_between(unique, counts, alpha=0.4)
    plt.plot(unique, counts)



# id_list, t50, t100, t200 = users_by_category('science/technology/health', users, class_threshold=0.5)
# scatter_by_sex(id_list, t200)
# scatter_by_age(id_list, t200)
# plot_categories_by_sex(id_stack, class_stack)
# plot_categories_by_age(id_stack, class_stack)
# plot_sentiment_by_sex(id_stack, polarity=sentiment_stack)
# plot_bot_by_categories(id_stack, class_stack, 0.5)
# print(get_users_by_topic(id_stack, topic100_stack, topid=67, topic_threshold=0.1))
