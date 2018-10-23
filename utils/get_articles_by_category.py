import newspaper
import json
import pymongo
import sys
sys.path.insert(0, '..')
import twitter_pipeline.article_scraper as news_scraper
import twitter_pipeline.category_classifier as category_classifier


def print_articles_stats():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    dbc = client["NewsAnalyzer"]

    cat = dbc.categories.distinct('category')
    print('Total articles: ' + str(dbc.articles.count()))
    for cc in cat:
        print(cc + ': ' + str(dbc.articles.find({'category': cc}).count()))


mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["NewsAnalyzer"]

fileSources = open('../data/sources.json').read()
sources = json.loads(fileSources)

category_list = db['categories'].distinct('category')

total_count = 0

for s in sources:
    insert_count = 0
    if s['url_category'] == 'True':
        print('Scraping ' + s['name'])
        news_source = newspaper.build('https://www.'+s['domain'][0], memoize_articles=False)
        print('Found articles: ' + str(news_source.size()))
        for article in news_source.articles:
            url = article.url
            url_hex = url.encode('utf-8').hex()

            # check if article is already in db
            if not db.articles.find_one({'_id': url_hex}):
                category = None
                splitted_url = url.split('/')

                if len(splitted_url) > int(s['url_category_position']):
                    category = splitted_url[int(s['url_category_position'])]

                if (category == 'news' or category == 'us') and len(splitted_url) > int(s['url_category_position'])+1:
                    category = url.split('/')[int(s['url_category_position']) + 1]

                db_category = db['categories'].find_one({'_id': category})

                for c in category_list:
                    if db_category and db_category['category'] == c:
                        news_data = news_scraper.scrape_news(url)
                        if news_data and len(news_data['keywords']) > 9:
                            news_data['category'] = c
                            news_data['category_aggregate'] = category_classifier.get_aggregated_category(c)
                            news_data['ground_truth'] = True
                            news_data['_id'] = url_hex[-64:]
                            if not db['articles'].find_one({'_id': news_data['_id']}):
                                db['articles'].insert_one(news_data)
                                insert_count = insert_count + 1
                                print('Inserted article in ' + c)

        total_count = total_count + insert_count
        print('---------- New articles categorized by ' + s['name'] + ' :' + str(insert_count))

print('----------------------- Total new articles: ' + str(total_count))
print_articles_stats()

