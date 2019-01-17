import json
import pymongo

db = None

try:
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client["NewsAnalyzer"]
except:
    print('No running instance of mongodb found in localhost. Please execute mongod command in terminal.')
    exit()

sources = json.loads(open('demo/sources.json').read())
categories = json.loads(open('demo/categories.json').read())

for s in sources:
    if not db.sources.find_one({'name': s['name']}):
        db['sources'].insert_one(s)
    else:
        print(s['name'] + ' already in database.')

for c in categories:
    if not db.categories.find_one({'_id': c['_id']}):
        db['categories'].insert_one(c)
    else:
        print(c['_id'] + ' already in database.')

src = list(db.sources.find({}, {"name": 1, "_id": 0}))
cat = list(db.categories.distinct('category'))

print('######################################')
print('')
print('Sources and categories setup completed.')
print('')
print('Sources:')
print(src)
print('')
print('Categories:')
print(cat)
print('')
print('######################################')


