import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
dbc = client["NewsAnalyzer"]

print('---------------------------------------------------------')
print('[USERS] Users count: ' + str(dbc.user.find().count()))
print('---------------------------------------------------------')
print('[TWEETS] Tweets count: ' + str(dbc.tweet.find().count()))

cat = ['world', 'politics', 'business', 'sports', 'entertainment/art', 'national/local', 'style/food/travel', 'science/technology/health']

all_art = {}
gt_art = {}
for cc in cat:
    all_art[cc] = dbc.articles.find({'category_aggregate': cc}).count()
    gt_art[cc] = dbc.articles.find({'ground_truth': True, 'category_aggregate': cc}).count()

gt_sort = sorted(gt_art.items(), key=lambda x: x[1], reverse=True)
all_sort = sorted(all_art.items(), key=lambda x: x[1], reverse=True)

print('---------------------------------------------------------')
print('[ARTICLES] All articles count: ' + str(dbc.articles.find().count()))
for k, v in all_sort:
    print(k + ': ' + str(v))
print('---------------------------------------------------------')
print('[ARTICLES] Ground truth articles count: ' + str(dbc.articles.find({'ground_truth': True}).count()))
for k, v in gt_sort:
    print(k + ': ' + str(v))
