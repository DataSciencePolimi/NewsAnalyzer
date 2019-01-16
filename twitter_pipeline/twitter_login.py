import tweepy
import json


# login to twitter you must have a file called credentialsTwitter.json with
# your consumer_key, consumer_secret, access_token, access_token_secret
def login(path=None):
    if not path:
        path = 'credentials.json'
    fileKeys = open(path).read()
    keys = json.loads(fileKeys)
    auth = tweepy.OAuthHandler(keys['consumer_key'], keys['consumer_secret'])
    auth.set_access_token(keys['access_token'], keys['access_token_secret'])
    twitter = tweepy.API(auth, wait_on_rate_limit=True)
    return twitter
