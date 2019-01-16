import newspaper
import pymongo
import twitter_pipeline.article_scraper as news_scraper

# PARAMETERS
MAX_SOURCE_ARTICLES = 20

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["NewsAnalyzer"]
sources = list(db.f_sources.find())

total_count = 0

for s in sources:
    insert_count = 0
    print('Scraping ' + s['domain'][0])

    news_source = newspaper.build('https://www.'+s['domain'][0], memoize_articles=False)
    print('Found articles: ' + str(news_source.size()))
    for article in news_source.articles:
        if insert_count > (MAX_SOURCE_ARTICLES - 1):
            break
        url = article.url
        url_hex = url.encode('utf-8').hex()

        # check if article is already in db
        if not db.t_articles.find_one({'_id': url_hex[-64:]}):
            news_data = news_scraper.scrape_news(url)
            if news_data:
                news_data['_id'] = url_hex[-64:]
                db['articles'].insert_one(news_data)
                insert_count = insert_count + 1
                print('Scraped news at url: ' + url)

        total_count = total_count + insert_count
    print('---------- New articles found by ' + s['domain'][0] + ' :' + str(insert_count))

print('----------------------- Total new articles: ' + str(total_count))



