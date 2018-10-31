import pymongo
import json
import requests
import urllib
import time

yandex_url = 'https://geocode-maps.yandex.ru/1.x/?lang=en&format=json&'


def updateMongoWithGeocoding(user, db):
    if not user['location'] or user['location'] == '':
        db.user.update({"_id": user["_id"]},
                       {"$set": {"y_geocode": None}})
        return

    req_url = yandex_url + urllib.parse.urlencode({'geocode': user['location']})
    response_geocode = requests.get(req_url)
    if response_geocode.status_code != 200:
        print(user['screen_name'] + ' Response error: ' + response_geocode.status_code)
        return
    response_json = json.loads(response_geocode.text)
    response_json = response_json['response']['GeoObjectCollection']

    if len(response_json['featureMember']) > 0:
        try:
            geocode_result = response_json['featureMember'][0]['GeoObject']
            coord = geocode_result['Point']['pos']
            address = geocode_result["metaDataProperty"]["GeocoderMetaData"]["Address"]
        except Exception as e:
            print(user['screen_name'] + ' Parsing error: ' + e)
            return

        data = {'coordinates': coord, 'address': address}
        db.user.update({"_id": user["_id"]},
                       {"$set": {"y_geocode": data}})
    else:
        db.user.update({"_id": user["_id"]},
                       {"$set": {"y_geocode": None}})
        return


def main():
    client = pymongo.MongoClient('localhost:27017')
    db = client.NewsAnalyzer

    counter_general = 0
    try:
        # for any user which has never been analyzed
        users = db.user.find({"y_geocode": {"$exists": False}})
        n_users = users.count()
        print('Processing ' + str(n_users) + ' missing users:')
        for res in users:
            try:
                counter_general = counter_general + 1

                updateMongoWithGeocoding(res, db)

                if divmod(counter_general, 10)[1] == 0:
                    print('Done ' + str(counter_general) + ' / ' + str(n_users))

            except Exception as exception:
                print('Oops!  An error occurred in loop.  Try again... line: ', exception)

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)


if __name__ == "__main__":
    main()
