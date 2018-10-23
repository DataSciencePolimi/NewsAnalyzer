import newspaper
import json
import pymongo
import twitter_pipeline.article_scraper as news_scraper

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["NewsAnalyzer"]

news_source = newspaper.build('https://abcnews.go.com/Health/', memoize_articles=False)
source_name = 'ABC News'

source_object = None
fileSources = open('../sources.json').read()
sources = json.loads(fileSources)

for s in sources:
    if s['name'] == source_name:
        source_object = s

if not source_object:
    print('ERROR: source not found.')

else:
    print('Found articles: ' + str(news_source.size()))
    for article in news_source.articles:
        url = article.url
        category = None

        if source_object['url_category'] == 'True':
            splitted_url = url.split('/')

            if len(splitted_url) > int(source_object['url_category_position']):
                category = splitted_url[int(source_object['url_category_position'])]

            if category == 'news':
                category = url.split('/')[int(source_object['url_category_position']) + 1]

            print(url)
            print(category)
'''
            if category and not db['categories'].find_one({'_id': category}):
                db['categories'].insert_one({'_id': category, 'example': url})
'''
