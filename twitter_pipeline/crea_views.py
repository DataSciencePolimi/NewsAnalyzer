import sys
import datetime
import pymongo
import csv
sys.path.append("/twitter_pipeline")
sys.path.append("/data")

"""
    This script generate few csv files containing aggregated statistics from db

"""

save_path = '../views/'

def main():
    print('Generating view files...')
    try:
        mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = mongo_client["NewsAnalyzer"]

        # VIEW UTENTI ------------------------------------
        csvData = [['Label', 'Field', 'Value', '% total', '% demographic']]

        user_count = db.user.find().count()  # count users
        csvData.append(['Users', 'Total count', user_count, '-', '-'])

        lang = db.user.distinct('lang')  # distinct languages
        for l in lang:
            l_count = db.user.find({'lang': l}).count()  # count user for lang
            csvData.append(['Language', l, l_count, percent(l_count, user_count), '-'])

        demo_count = db.user.find({"t_f_api": 4}).count()
        demo_uknown = user_count - demo_count
        sex_m = db.user.find({"t_gender": 2}).count()  # sex distribution
        sex_f = db.user.find({"t_gender": 1}).count()
        csvData.append(['Demographics', 'Unknown', demo_uknown, percent(demo_uknown, user_count), '-'])
        csvData.append(['Demographics', 'Processed', user_count - demo_uknown, percent(user_count - demo_uknown, user_count), '-'])
        csvData.append(['Sex', 'Male', sex_m, percent(sex_m, user_count), percent(sex_m, demo_count)])
        csvData.append(['Sex', 'Female', sex_f, percent(sex_f, user_count), percent(sex_f, demo_count)])

        ethn = db.user.distinct('t_eth')  # distinct ethnicity
        for e in ethn:
            eth_count = db.user.find({'t_eth': e}).count()  # count user for ethnicity
            csvData.append(['Etnicity', e, eth_count, percent(eth_count, user_count), percent(eth_count, demo_count)])

        age_lt_20 = db.user.find({'t_age': {'$lt': 20}}).count()
        age_lt_30 = db.user.find({'t_age': {'$gt': 19, '$lt': 30}}).count()
        age_lt_40 = db.user.find({'t_age': {'$gt': 29, '$lt': 40}}).count()
        age_lt_50 = db.user.find({'t_age': {'$gt': 39, '$lt': 50}}).count()
        age_lt_60 = db.user.find({'t_age': {'$gt': 49, '$lt': 60}}).count()
        age_gt_60 = db.user.find({'t_age': {'$gt': 59}}).count()
        csvData.append(['Age', '< 20', age_lt_20, percent(age_lt_20, user_count), percent(age_lt_20, demo_count)])
        csvData.append(['Age', '20 - 30', age_lt_30, percent(age_lt_30, user_count), percent(age_lt_30, demo_count)])
        csvData.append(['Age', '30 - 40', age_lt_40, percent(age_lt_40, user_count), percent(age_lt_40, demo_count)])
        csvData.append(['Age', '40 - 50', age_lt_50, percent(age_lt_50, user_count), percent(age_lt_50, demo_count)])
        csvData.append(['Age', '50 - 60', age_lt_60, percent(age_lt_60, user_count), percent(age_lt_60, demo_count)])
        csvData.append(['Age', '> 60', age_gt_60, percent(age_gt_60, user_count), percent(age_gt_60, demo_count)])

        with open(save_path + 'users.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            writer.writerows(csvData)

        csvFile.close()
        print('DONE users.csv')

        # VIEW TWEET ------------------------------------
        csvData = [['Label', 'Field', 'Value', '% on total']]
        tweet_count = db.tweet.find().count()  # count tweets
        csvData.append(['Tweet', 'Total count', tweet_count, '-'])
        RT_count = db.tweet.find({'RT': True}).count()  # count retweets
        csvData.append(['Retweet', 'Count', RT_count, percent(RT_count, tweet_count)])
        sent_polarity_POS = db.tweet.find({'sentiment_polarity': {'$gt': 0.1, '$lt': 0.5}}).count()
        sent_polarity_VPOS = db.tweet.find({'sentiment_polarity': {'$gt': 0.5}}).count()
        sent_polarity_NEG = db.tweet.find({'sentiment_polarity': {'$gt': -0.5, '$lt': -0.1}}).count()
        sent_polarity_VNEG = db.tweet.find({'sentiment_polarity': {'$lt': -0.5}}).count()
        sent_polarity_NEU = db.tweet.find({'sentiment_polarity': {'$gt': -0.1, '$lt': 0.1}}).count()
        csvData.append(
            ['Sentiment polarity', 'Very positive', sent_polarity_VPOS, percent(sent_polarity_VPOS, tweet_count)])
        csvData.append(
            ['Sentiment polarity', 'Positive', sent_polarity_POS, percent(sent_polarity_POS, tweet_count)])
        csvData.append(
            ['Sentiment polarity', 'Neutral', sent_polarity_NEU, percent(sent_polarity_NEU, tweet_count)])
        csvData.append(
            ['Sentiment polarity', 'Negative', sent_polarity_NEG, percent(sent_polarity_NEG, tweet_count)])
        csvData.append(
            ['Sentiment polarity', 'Very negative', sent_polarity_VNEG, percent(sent_polarity_VNEG, tweet_count)])

        sent_subj_L = db.tweet.find({'sentiment_subjectivity': {'$lt': 0.3}}).count()
        sent_subj_M = db.tweet.find({'sentiment_subjectivity': {'$gt': 0.3, '$lt': 0.6}}).count()
        sent_subj_H = db.tweet.find({'sentiment_subjectivity': {'$gt': 0.6}}).count()
        csvData.append(
            ['Sentiment subjectivity', 'Low', sent_subj_L, percent(sent_subj_L, tweet_count)])
        csvData.append(
            ['Sentiment subjectivity', 'Medium', sent_subj_M, percent(sent_subj_M, tweet_count)])
        csvData.append(
            ['Sentiment subjectivity', 'High', sent_subj_H, percent(sent_subj_H, tweet_count)])

        with open(save_path + 'tweets.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            writer.writerows(csvData)

        csvFile.close()
        print('DONE tweets.csv')

        # VIEW ARTICOLI ------------------------------------
        csvData = [['Label', 'Field', 'Value', '% on total']]
        art_count = db.articles.find().count()  # count articles
        art_gt = db.articles.find({'ground_truth': True}).count()
        art_pred = art_count - art_gt
        art_pip = db.articles.find({'pipelined': True}).count()
        csvData.append(['Articles', 'Total count', art_count, '-'])
        csvData.append(['Pipeline', 'Pipelined count', art_pip, percent(art_pip, art_count)])
        csvData.append(['Classifier', 'Ground truth', art_gt, percent(art_gt, art_count)])
        csvData.append(['Classifier', 'Predicted', art_pred, percent(art_pred, art_count)])

        categories = ['world', 'politics', 'business', 'sports', 'entertainment/art', 'national/local',
                      'style/food/travel', 'science/technology/health']

        for c in categories:
            cat_count = db.articles.find({'category_aggregate': c}).count()
            csvData.append(['Category', c, cat_count, percent(cat_count, art_count)])

        art_by_source = {}
        for s in db.sources.find():
            art_by_source[s['name']] = db.articles.find({'source_name': s['name']}).count()
        art_by_source_sort = sorted(art_by_source.items(), key=lambda x: x[1], reverse=True)
        s_counter = 0
        a_counter = 0
        for k, v in art_by_source_sort:
            if s_counter == 15:
                csvData.append(['Source', 'other', art_count-a_counter, percent(art_count-a_counter, art_count)])
                break
            csvData.append(['Source', k, v, percent(v, art_count)])
            a_counter = a_counter + v
            s_counter = s_counter + 1

        with open(save_path + 'articles.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            writer.writerows(csvData)

        csvFile.close()
        print('DONE articles.csv')

        # VIEW SORGENTI / CATEGORIE ------------------------------------
        csvData = [['Source name', 'Articles count', 'world', 'politics', 'business', 'sports', 'entertainment/art', 'national/local',
                      'style/food/travel', 'science/technology/health']]
        sources = db.articles.aggregate([
            {"$group": {'_id': "$source_name", 'count': {'$sum': 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 20}
        ])
        for s in sources:
            row = [s['_id'], s['count']]
            for c in categories:
                row.append(percent(db.articles.find({'source_name': s['_id'], 'category_aggregate': c}).count(), s['count']))
            csvData.append(row)

        with open(save_path + 'sources.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            writer.writerows(csvData)

        csvFile.close()
        print('DONE sources.csv')

        # VIEW UTENTI / TWEET / SORGENTI / CATEGORIE ------------------------------------
        csvData = [['Id_User', 'Screen name', 'Tweet count', 'average sentiment',
                    'world', 'politics', 'business', 'sports', 'entertainment/art',
                    'national/local', 'style/food/travel', 'science/technology/health',
                    'top_source_1', 'top_source_2', 'top_source_3',
                     ]
                    ]
        top_users = db.tweet.aggregate([
            {"$group": {'_id': "$id_user", 'count': {'$sum': 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 40}
        ])
        for u in top_users:
            tu_screen_name = db.user.find({'_id': int(u['_id'])}).next()['screen_name']
            tu_tw_count = db.tweet.find({'id_user': u['_id']}).count()
            tu_average_sent = db.tweet.aggregate([
                    { "$match": { "id_user": u['_id'] } },
                    { "$group": { "_id": None, 'average': {"$avg": "$sentiment_polarity" } } }]).next()['average']
            tu_ref_art = []
            for a in db.tweet.find({'id_user': u['_id']}):
                tu_ref_art.append(a['article'])
            tu_cat_distribution = db.articles.aggregate([
                {'$match': {'_id': {'$in': tu_ref_art}}},
                {'$group': { '_id': '$category_aggregate', 'count': {'$sum': 1}}},
                {"$sort": {"count": -1}}
            ])

            tu_cat_dict = {}
            for k in tu_cat_distribution:
                tu_cat_dict[k['_id']] = k['count']
            for c in categories:
                if c not in tu_cat_dict:
                    tu_cat_dict[c] = 0

            tu_source_distribution = db.articles.aggregate([
                {'$match': {'_id': {'$in': tu_ref_art}}},
                {'$group': {'_id': '$source_name', 'count': {'$sum': 1}}},
                {"$sort": {"count": -1}},
                {'$limit': 3}
            ])
            tu_source_top = ['-', '-', '-']
            tu_source_index = 0
            for s in tu_source_distribution:
                tu_source_top[tu_source_index] = str(s['_id']) + ': ' + percent(s['count'], tu_tw_count)
                tu_source_index = tu_source_index + 1

            row = [u['_id'], tu_screen_name, tu_tw_count, round(tu_average_sent, 3),
                   percent(tu_cat_dict['world'], tu_tw_count),
                   percent(tu_cat_dict['politics'], tu_tw_count),
                   percent(tu_cat_dict['business'], tu_tw_count),
                   percent(tu_cat_dict['sports'], tu_tw_count),
                   percent(tu_cat_dict['entertainment/art'], tu_tw_count),
                   percent(tu_cat_dict['national/local'], tu_tw_count),
                   percent(tu_cat_dict['style/food/travel'], tu_tw_count),
                   percent(tu_cat_dict['science/technology/health'], tu_tw_count),
                   tu_source_top[0],
                   tu_source_top[1],
                   tu_source_top[2]
                   ]
            csvData.append(row)

        with open(save_path + 'top_users.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            writer.writerows(csvData)

        csvFile.close()
        print('DONE top_users.csv')

        # VIEW TWEET / ARTICOLI ------------------------------------
        csvData = [['Article URL', 'Tweet count', 'Source', 'Category']]
        tw_by_art = db.tweet.aggregate([
            {"$group": {'_id': "$article", 'count': {'$sum': 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 40}
        ])
        for tw_group in tw_by_art:
            article = db.articles.find_one({'_id': tw_group['_id']})
            if article:
                csvData.append([article['url'], tw_group['count'], article['source_name'], article['category_aggregate']])

        with open(save_path + 'top_articles.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            writer.writerows(csvData)

        csvFile.close()
        print('DONE top_articles.csv')

    except Exception as exception:
        print('Oops!  An error occurred in loop.  Try again... error: ' + str(exception))
        return


def percent(fraction, total):
    # return str(round((fraction / total) * 100, 2)) + '%'
    return str(round((fraction / total) * 100, 2))


if __name__ == "__main__":
    timeStart = datetime.datetime.now()
    main()
    timeEnd = datetime.datetime.now()
    delta = timeEnd - timeStart
    print('Executed in ' + str(int(delta.total_seconds())) + 's')

