from urllib.parse import urlparse
from http import client
import re
import requests
import json


def unshorten_url(url, count=0):
    try:
        parsed = urlparse(url)
        h = client.HTTPConnection(parsed.netloc)
        resource = parsed.path
        if parsed.query != "":
            resource += "?" + parsed.query
        h.request('HEAD', resource)
        response = h.getresponse()
        if response.status // 100 == 3 and response.getheader('Location') and count < 3:
            return unshorten_url(response.getheader('Location'), count+1)  # changed to process chains of short urls
        else:
            return url
    except:
        return url


def extract_raw_link(text):
    # look for http links
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    splitted_urls = []
    for u in urls:
        splitted_urls.extend(u.split(' '))
    if len(splitted_urls) > 0:
        return splitted_urls[0]
    else:
        return None


def extract_known_sources(tweet, sources):
    try:
        u = tweet['news_url']
        for s in sources:
            for d in s['domain']:
                if u and d in u:
                    tweet['news_source'] = s['name']
    except KeyError:
        return tweet
    return tweet

