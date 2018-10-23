import tweepy
import json
import sys
import datetime
import csv
import sys
import getopt


# list of users who post the news
def get_users_from_news(news, twitter, N=None):
    users = []

    for status in tweepy.Cursor(twitter.search, news).items():
        if N is not None and len(users) > N:
            break
        elif status.user.lang == 'en':
            users.append(status.user.screen_name)
    return users


