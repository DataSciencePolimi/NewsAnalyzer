import newspaper
import pymongo
import pickle
from pymongo import errors
import twitter_pipeline.article_scraper as news_scraper
import twitter_pipeline.enrich_news_tweet as enricher
import category_classifier.category_classifier as category_classifier

'''
This script process the list of news sources, 
scraping and classifying a set of news articles to be used as warm start input for the collection pipeline.

'''

# PARAMETERS
# max number of articles to be scraped from each source
MAX_ARTICLES_BY_SOURCE = 20

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["NewsAnalyzer"]
sources = list(db.sources.find())

models = {}
try:
    # load the model from disk
    models = {}
    models['model'] = pickle.load(open('../category_classifier/models/classifier.sav', 'rb'))
    models['countvect'] = pickle.load(open('../category_classifier/models/countvect.sav', 'rb'))
    models['labelencoder'] = pickle.load(open('../category_classifier/models/labelencoder.sav', 'rb'))
except:
    pass

total_count = 0

for s in sources:
    insert_count = 0
    print('Scraping ' + s['domain'][0])

    news_source = newspaper.build('https://www.'+s['domain'][0], memoize_articles=False)
    print('Found articles: ' + str(news_source.size()))
    for article in news_source.articles:
        if insert_count > (MAX_ARTICLES_BY_SOURCE - 1):
            break
        url = article.url
        url_hex = url.encode('utf-8').hex()

        # check if article is already in db
        if not db.t_articles.find_one({'_id': url_hex[-64:]}):
            news_data = news_scraper.scrape_news(url)
            if news_data:
                news_data['_id'] = url_hex[-64:]
                category = enricher.extract_category_from_url(url, s['name'], db)
                ground_truth = True
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

                category_aggregate = enricher.get_aggregated_category(category)

                news_data['category'] = category
                news_data['category_aggregate'] = category_aggregate
                news_data['category_probability'] = category_prob
                news_data['ground_truth'] = ground_truth

                try:
                    db['articles'].insert_one(news_data)
                except pymongo.errors.DuplicateKeyError:
                    continue
                insert_count = insert_count + 1
                print('Scraped news at url: ' + url)

        total_count = total_count + insert_count
    print('---------- New articles found by ' + s['domain'][0] + ' :' + str(insert_count))

print('----------------------- Total new articles: ' + str(total_count))



