import json
import pickle
from textblob import TextBlob
import category_classifier.category_classifier as category_classifier

'''
This script enrich tweets by adding a category from url (if present) and a set of keywords.

'''

try:
    # load the model from disk
    models = {}
    models['model'] = pickle.load(open('models/classifier.sav', 'rb'))
    models['countvect'] = pickle.load(open('models/countvect.sav', 'rb'))
    # models['tfidfvect'] = pickle.load(open('models/tfidfvect.sav', 'rb'))
    models['labelencoder'] = pickle.load(open('models/labelencoder.sav', 'rb'))
except:
    pass


def enrich_news_tweets(input_file, sources_file):
    fileTweets = open(input_file).read()
    tweets = json.loads(fileTweets)

    for user in tweets:
        tw_count = 0
        print('Processing ' + user)
        for tw in tweets[user]:
            tw = process_tweet(tw, sources_file)
            tw_count = tw_count+1
            print(str(tw_count) + ' / ' + str(len(tweets[user])))
    return tweets


def save_tweets(data, name_file):
    j = json.dumps(data)
    file = open(name_file, 'w')
    file.write(j)
    return


def process_tweet(tweet, news_data, db):
    url = tweet['news_url']
    source_name = tweet['news_source']
    # news_data = news_scraper.scrape_news(url)
    if not news_data:
        return None
    ground_truth = True
    category = extract_category_from_url(url, source_name, db)
    category_prob = 1
    if not category:
        ground_truth = False
        pred = category_classifier.predict(news_data['keywords'], models)
        if pred:
            category = pred['class']
            category_prob = pred['probability']
        else:
            category = None
            category_prob = None

    news_data['category'] = category
    news_data['category_aggregate'] = category_classifier.get_aggregated_category(category)
    news_data['category_probability'] = category_prob
    news_data['ground_truth'] = ground_truth
    # insert article
    if news_data and not db['articles'].find_one({'_id': news_data['_id']}):
        db['articles'].insert_one(news_data)

    # set article reference in tweet
    tweet['article'] = news_data['_id']
    # run sentiment analysis on text
    sa = TextBlob(tweet['text'])
    tweet['sentiment_polarity'] = sa.sentiment.polarity
    tweet['sentiment_subjectivity'] = sa.sentiment.subjectivity
    return tweet


def extract_category_from_url(url, source_name, db):
    category = None
    sources = db.sources.find()
    for s in sources:
        if source_name == s['name'] and s['url_category'] == 'True':
            splitted_url = url.split('/')
            if len(splitted_url) > int(s['url_category_position']):
                category = splitted_url[int(s['url_category_position'])]
            if (category == 'news' or category == 'us') and len(splitted_url) > int(s['url_category_position']) + 1:
                category = url.split('/')[int(s['url_category_position']) + 1]
            return get_consolidated_category(category, db)
    return None


def get_consolidated_category(raw_category, db):
    category_row = db.categories.find_one({'_id': raw_category})
    if category_row:
        return category_row['category']
    else:
        return None
