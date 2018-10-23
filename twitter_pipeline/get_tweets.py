import tweepy
import time
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


def user_tweets_to_mongo(account, twitter, mongo, sources, N=3000):
    try:
        if 'fully_scraped' in mongo.user.find_one({'screen_name': account}):
            return {'total': 0, 'useful': 0}
    except:
        pass
    max_number = 3200
    max_per_request = 200
    languages = ['en']
    user_tweets = []
    if N > max_number:
        N = max_number
        iteration = 16
        last = 0
    else:
        iteration, last = divmod(N, max_per_request)

    try:
        user_timeline = twitter.user_timeline(screen_name=account, count=1, include_rts=True, tweet_mode="extended")
        if user_timeline:
            for i in range(iteration + 1):
                lastTweetId = int(user_timeline[-1].id_str)
                user_timeline = twitter.user_timeline(screen_name=account, max_id=lastTweetId, count=max_per_request,
                                                      include_rts=True, tweet_mode="extended")
                for tweets in user_timeline:
                    if not tweets.lang:
                        tweets.lang = detect(tweets.text.replace("\n", " "))
                    if tweets.lang in languages:
                        d = {'id_user': tweets.user.id_str, 'screen_name': tweets.user.screen_name.lower(),
                             'text': tweets.full_text, 'lang': tweets.lang, 'favourite_count': tweets.favorite_count,
                             'retweet_count': tweets.retweet_count,
                             'create_at': tweets.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                             'mentions': tweets.entities['user_mentions'], '_id': tweets.id_str,
                             'coordinates': tweets.coordinates, 'entities': tweets.entities, 'RT': False}
                        if hasattr(tweets, 'retweeted_status'):
                            d['RT'] = True
                            d['RT_id'] = tweets.retweeted_status.id_str
                            d['RT_entities'] = tweets.retweeted_status.entities
                        user_tweets.append(d)
                if i != iteration:
                    user_tweets = user_tweets[:len(user_tweets) - 1]
                if len(user_timeline) < max_per_request:
                    break
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

    user_tweets = filtered_tweets[:N]
    n_useful = len(user_tweets)

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

    # download articles and store tweet + article
    for t in user_tweets:
        t = enricher.process_tweet(t, articles[t['_id']], mongo)
        if t and not mongo['tweet'].find_one({'_id': t['_id']}):
            mongo['tweet'].insert_one(t)
        mongo.user.update({"screen_name": account},
                          {"$set": {"fully_scraped": True}})
    return {'total': n_total, 'useful': n_useful}

