import getopt
import requests
import pymongo
import datetime
import twitter_pipeline.category_classifier as category_classifier
from newspaper import Article

"""
    This script scrapes and article from its url

    Requires:
    ----------
    news_url to be scraped

"""

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["NewsAnalyzer"]
domain_dictionary = {}
for s in db.sources.find():
    for d in s['domain']:
        domain_dictionary[d] = s['name']


def scrape_news(news_url, language='en', nlp=True):
    try:
        news = Article(news_url, language=language, MAX_KEYWORDS=30, MAX_AUTHORS=1, fetch_images=False)
        # download the document
        news.download()
        # parse text content
        if news.download_state != 2:
            return None
        news.parse()
        # run nlp for keywords
        if nlp:
            news.nlp()

        news_data = {}

        news_data['url'] = news.url.split('?')[0]
        news_data['_id'] = news_data['url'].encode('utf-8').hex()[-64:]
        news_data['title'] = news.title
        news_data['authors'] = news.authors
        news_data['text'] = news.text
        news_data['language'] = news.meta_lang
        news_data['keywords'] = category_classifier.transform_keywords_array(news.keywords)
        news_data['tags'] = category_classifier.transform_keywords_array(list(news.tags))
        news_data['source'] = news.source_url
        news_data['source_name'] = None
        news_data['publish_date'] = str(news.publish_date)
        news_data['scrape_date'] = str(datetime.datetime.now())
        news_data['pipelined'] = False

        if not validate_text(news_data['text']) or news_data['title'] == '':
            return None

        # extract source name
        for k in domain_dictionary:
            if k in news_data['source']:
                news_data['source_name'] = domain_dictionary[k]
                break

        # try to extract source from redirect link
        if news_data['source_name'] is None:
            r = requests.get(news.url)
            if r and r.status_code == 200:
                news_data['url'] = r.url.split('?')[0]
                news_data['_id'] = news_data['url'].encode('utf-8').hex()[-64:]
                # extract source name
                for k in domain_dictionary:
                    if k in news_data['url']:
                        news_data['source'] = k
                        news_data['source_name'] = domain_dictionary[k]
                        break

        return news_data

    except:
        return None


black_list = [
        'Please enable cookies on your web browser',
        'Terms of Service Violation',
        'Unfortunately, our website is currently unavailable',
        'By choosing “I agree” below, you agree that NPR',
        'About Your Privacy on this Site',
        'Chat with us in Facebook Messenger',
        'What term do you want to search? Search with google',
        'This site is not available in your region',
        'We can’t find a newsday subscription associated',
        'Desktop notifications are on',
        'Already a subscriber?',
        'This demonstration page uses our WeatherBlox',
        'NYTimes.com no longer supports Internet Explorer',
        'Tap here to turn on desktop notifications',
        'Warning! This page uses Javascript.'
    ]


def validate_text(text):
    if text == '':
        return False
    for b in black_list:
        if b in text:
            return False
    return True
