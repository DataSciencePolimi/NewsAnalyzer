import pymongo
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA
import pprint as pp


"""
    This script filter user vectorized file by extracting republicans / democrats users

    Requires:
    ----------
    v_users : file of vectorized users

"""

def get_republican_democratic_clusters(v_users):
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client["NewsAnalyzer"]

    rep_keywords = ['#maga', '#kag', '#trump2020' '#americafirst', '#trumptrain', '#armyoftrump', '#votegop2018', '#backtheblue', '#redwaverising', '#trumpismypresident' ]
    dem_keywords = ['#bluewave', '#theresistance', '#votebluetosaveamerica', '#notmypresident', '#voteblue', '#uniteblue', '#nevertrump']
    exrep_keywords = ['ex-gop', '#exgop']

    # extract democratic and republican users by hashtags in user description
    rep_users = []
    dem_users = []

    for u in db.user.find():
        if any(word in u['description'].lower() for word in rep_keywords):
            rep_users.append(u['_id'])
        elif any(word in u['description'].lower() for word in dem_keywords):
            dem_users.append(u['_id'])

    print('Democratic users: ' + str(len(dem_users)))
    print('Republican users: ' + str(len(rep_users)))

    # pickle.dump(rep_users, open('data/rep_users.sav', 'wb'))
    # pickle.dump(dem_users, open('data/dem_users.sav', 'wb'))

    # get vectorized representation of users
    v_rep = []
    v_dem = []
    for u in v_users:
        if u['user_id'] in rep_users:
            v_rep.append(u)
        elif u['user_id'] in dem_users:
            v_dem.append(u)

    return v_rep, v_dem
