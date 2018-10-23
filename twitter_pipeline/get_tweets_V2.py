import tweepy
import time
from datetime import datetime
from langdetect import detect
import threading
from threading import Thread
import twitter_pipeline.news_tweet_filter as url_filter
import twitter_pipeline.enrich_news_tweet as enricher
import twitter_pipeline.article_scraper as news_scraper

# Definizione del lock
threadLock = threading.Lock()


class ArticleThread(Thread):
    def __init__(self, nome, tweet, articles):
        Thread.__init__(self)
        self.nome = nome
        self.tweet = tweet
        self.articles = articles

    def run(self):
        url = self.tweet['news_url']
        news_data = news_scraper.scrape_news(url)

        # Acquisizione del lock
        threadLock.acquire()
        self.articles[self.tweet['_id']] = news_data
        # Rilascio del lock
        threadLock.release()


def user_tweets_to_mongo(account, twitter, mongo, sources):
    user_id = mongo.user.find_one({'screen_name': account})['_id']
    languages = ['en']
    user_tweets = []

    # tw_by_month = {2017: {}, 2018: {}}
    # for i in range(1, 13):
        # tw_by_month[2017][i] = 0
        # tw_by_month[2018][i] = 0

    try:
        for status in tweepy.Cursor(twitter.user_timeline, screen_name=account, include_rts=True, tweet_mode="extended").items():
            # tweet too old
            if status.created_at.year < 2018 and status.created_at.month < 8:
                break
            if not status.lang:
                status.lang = detect(status.text.replace("\n", " "))
            if status.lang in languages:
                d = {'id_user': status.user.id_str, 'screen_name': status.user.screen_name.lower(),
                     'text': status.full_text, 'lang': status.lang, 'favourite_count': status.favorite_count,
                     'retweet_count': status.retweet_count,
                     'create_at': status.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                     'mentions': status.entities['user_mentions'], '_id': status.id_str,
                     'coordinates': status.coordinates, 'entities': status.entities, 'RT': False}
                if hasattr(status, 'retweeted_status'):
                    d['RT'] = True
                    d['RT_id'] = status.retweeted_status.id_str
                    d['RT_entities'] = status.retweeted_status.entities
                user_tweets.append(d)

    except tweepy.RateLimitError:
        print('TWITTER LIMIT REACHED: sleep for 15 mins')
        time.sleep(15 * 60)
    except tweepy.TweepError:
        print('TWEEPY GENERIC ERROR: pass')
        pass

    n_total = len(user_tweets)

    tweets_with_link = []

    for i in range(0, len(user_tweets)):
        if not mongo['tweet'].find_one({'_id': user_tweets[i]['_id']}):
            link = None
            if len(user_tweets[i]['entities']['urls']) > 0:
                link = user_tweets[i]['entities']['urls'][0]['expanded_url']
            elif 'RT_entities' in user_tweets[i] and len(user_tweets[i]['RT_entities']['urls']) > 0:
                link = user_tweets[i]['RT_entities']['urls'][0]['expanded_url']
            user_tweets[i]['news_url'] = link
            if link:
                tweets_with_link.append(user_tweets[i])

    filtered_tweets = []
    for t in tweets_with_link:
        t = url_filter.extract_known_sources(t, sources)
        if 'news_source' in t:
            filtered_tweets.append(t)

    user_tweets = filtered_tweets
    n_useful = len(user_tweets)

#    for t in user_tweets:
#        date = datetime.strptime(t['create_at'], '%Y-%m-%d %H:%M:%S')
#        tw_by_month[date.year][date.month] += 1

    # download articles and store tweet + article
    # TODO limit the size of the thread pool and iterate on pools
    thread_pool = []
    articles = {}
    index = 0
    for t in user_tweets:
        thread_pool.append(ArticleThread(index, t, articles))

    for th in thread_pool:
        th.start()
    for th in thread_pool:
        th.join()

    n_useful += mongo.tweet.find({'id_user': user_id}).count()
    # download articles and store tweet + article
    for t in user_tweets:
        t = enricher.process_tweet(t, articles[t['_id']], mongo)
        if t and not mongo['tweet'].find_one({'_id': t['_id']}):
            mongo['tweet'].insert_one(t)

    # update user tweet counts
    mongo.user.update({"_id": user_id},
                      {"$set": {"frame_total_count": n_total, "frame_useful_count": n_useful, "fully_scraped": True}})
    return {'total': n_total, 'useful': n_useful}

