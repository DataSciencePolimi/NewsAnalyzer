import json
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["NewsAnalyzer"]

sources = json.load(open('demo/sources.json').read())
categories = json.load(open('demo/categories.json').read())

for s in sources:
    db['sources'].insert_one(s)

for c in categories:
    db['categories'].insert_one(c)


