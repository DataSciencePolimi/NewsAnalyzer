import pymongo
import twitter_pipeline.article_scraper as scraper


def clear_blacklisted_text(db):
    articles = db.articles.find()
    for a in articles:
        try:
            if not scraper.validate_text(a['text']):
                db.articles.delete_one({'_id': a['_id']})
        except:
            pass


def remove_duplicates(db):
    pipeline = [{"$group": {"_id": "$text", "count": {"$sum": 1}}},
                {"$match": {"_id": {"$ne": None}, "count": {"$gt": 1}}},
                {"$sort": {"count": -1}},
                {"$project": {"name": "$_id", "_id": 0}}
                ]
    # get all duplicate texts in article collection
    duplicate_texts = list(db.articles.aggregate(pipeline, allowDiskUse=True))
    for t in duplicate_texts:
        # query all articles with text t
        dup_list = db.articles.find({'text': t['name']})
        if dup_list.count() > 1:
            for a in dup_list:
                ref_tweets = db.tweet.find({'article': a['_id']})
                for tw in ref_tweets:
                    db.tweet.delete_one({"_id": tw["_id"]})
                db.articles.delete_one({'_id': a['_id']})


mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["NewsAnalyzer"]

remove_duplicates(db)
