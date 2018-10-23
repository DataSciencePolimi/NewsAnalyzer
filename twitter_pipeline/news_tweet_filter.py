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


def expand(query):
    BATCH_SIZE = 10
    result = []
    if not query or query == '':
        return result
    fileKeys = open('credentialsTwitter.json').read()
    keys = json.loads(fileKeys)

    # divide request in max 50 count batches
    qlist = query.split('***')
    size = len(qlist)
    iteration, rest = divmod(size, BATCH_SIZE)
    if rest > 0:
        iteration = iteration + 1

    for j in range(0, iteration):
        q = '***'.join(qlist[j*BATCH_SIZE:(j+1)*BATCH_SIZE])
        print('[EXPANDER] Request sent ' + str(j))
        r = requests.get(keys['urlex_endpoint']+q)
        print('[EXPANDER] done')
        try:
            response = r.json()
            for i in response:
                result.append(i['longurl'])
        except:
            print('[EXPANDER] Exception')
            pass
    return result


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

