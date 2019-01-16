import botometer
import json
import pymongo
from datetime import datetime
import pprint as pp

"""
    This script evaluates bot probability for a given twitter user

    Requires:
    ----------
    user_id OR screen_name

"""

def bot_check_by_id(userId):
    fileKeys = open('../credentials/credentialsTwitter.json').read()
    keys = json.loads(fileKeys)

    mashape_key = keys['mashape_key']
    twitter_app_auth = {
        'consumer_key': keys['consumer_key'],
        'consumer_secret': keys['consumer_secret'],
        'access_token': keys['access_token'],
        'access_token_secret': keys['access_token_secret'],
      }
    bom = botometer.Botometer(wait_on_ratelimit=True,
                              mashape_key=mashape_key,
                              **twitter_app_auth)

    # Check a single account by id
    result = bom.check_account(userId)
    return result['scores']


def bot_check_by_screenname(userId):
    fileKeys = open('../credentials/credentialsTwitter.json').read()
    keys = json.loads(fileKeys)

    mashape_key = keys['mashape_key']
    twitter_app_auth = {
        'consumer_key': keys['consumer_key'],
        'consumer_secret': keys['consumer_secret'],
        'access_token': keys['access_token'],
        'access_token_secret': keys['access_token_secret'],
    }
    bom = botometer.Botometer(wait_on_ratelimit=True,
                              mashape_key=mashape_key,
                              **twitter_app_auth)

    # Check a single account by id
    result = bom.check_account('@'+userId)
    return result['scores']


def main():
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client["NewsAnalyzer"]
    counter = 0

    try:
        for u in db.user.find({"bot_score": {"$exists": False}}):
            counter += 1
            if divmod(counter, 10)[1] == 0:
                print('Done users: ' + str(counter))
            try:
                scores = bot_check_by_id(u['_id'])
                db.user.update({"_id": u["_id"]}, {"$set": {"bot_score": scores}})
            except Exception as e:
                print(e)
                db.user.update({"_id": u["_id"]}, {"$set": {"bot_score": None}})
                continue
    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    timeStart = datetime.now()
    main()
    timeEnd = datetime.now()
    delta = timeEnd - timeStart
    print('Executed in ' + str(int(delta.total_seconds())) + 's')

