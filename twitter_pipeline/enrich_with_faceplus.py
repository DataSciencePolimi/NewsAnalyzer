import pymongo
import requests
import json

'''
IMPORTANT: Face++ is no more available in EU. To use this script you must connect through an extra-EU VPN

t_f_api FIELD
    - not exist : user has not been analyzed
    - 0 : user has default profile image
    - 1 : unable to detect single face
    - 2 : user has no profile image
    - 3 : no faces detected
    - 4 : face has been correctly analyzed
    
t_gender
    - 1 : female
    - 2 : male
    - -1: unknown


'''

client = pymongo.MongoClient('localhost:27017')
db = client.NewsAnalyzer

fileKeys = open('../credentials/credentialsTwitter.json').read()
keys = json.loads(fileKeys)

faceplusplusurldetect = "https://api-us.faceplusplus.com/facepp/v3/detect?api_key="+keys['faceplus_key']+"&api_secret="+keys['faceplus_secret']
faceplusplusurlanalyse = "https://api-us.faceplusplus.com/facepp/v3/face/analyze?api_key="+keys['faceplus_key']+"&api_secret="+keys['faceplus_secret']+"&return_attributes=gender,age,ethnicity"


def isTweetEligibleForFaceOperation(row):
    isEligible = False

    try:
        keys = row.keys()
        if "t_gender" in keys or "t_f_api" in keys or "api_res" in keys:
            return isEligible

        if "profile_image_url" not in row:
            return isEligible

        if row["profile_image_url"] is None or row["profile_image_url"] == "":
            # this user has not a profile image.. we add this info to db to prevent repetitive controls
            db.user.update({"_id": row["_id"]}, {"$set": {"t_f_api": 2}})
            return isEligible

        isEligible = True
        return isEligible
    except Exception as exception:
        print('Oops!  An error occurred in loop.  Try again... line: ' + str(row), exception)


def imageUrlToSend(row):
    img = ""
    try:
        img = row["profile_image_url"]
        if "default_profile" in img:
            # print(row['screen_name'] + " : this profile has default egg picture")
            db.user.update({"_id": row["_id"]}, {"$set": {"t_f_api": 0}})
        elif "_normal." in img:
            img = img.replace("_normal.", ".")
        else:
            print(row['screen_name'] + " : unexpected photo url pattern")

    except Exception as exception:
        print('Oops!  An error occurred in loop.  Try again... line: ' + str(row), exception)

    return img


def updateMongoWithImageDemographics(row):
    try:
        img = imageUrlToSend(row)
        if img is None or img == "":
            return

        newurldetect = faceplusplusurldetect + "&image_url=" + img
        responsedetect = requests.post(newurldetect)
        if responsedetect.status_code != 200:
            # print(row['screen_name'] + ' Response error: ' + responsedetect.status_code)
            return
        responsejsondetect = responsedetect.json()

        if "faces" not in responsejsondetect:
            # print(row['screen_name'] + " : this profile's picture is not convenient for faceplusplus. img: " + str(img))
            db.user.update({"_id": row["_id"]}, {"$set": {"t_f_api": 3}})
            return

        if len(responsejsondetect["faces"]) == 1:
            # when the object type is dict, you should use that kind of form
            faceobjects = responsejsondetect["faces"]
            faceobject = faceobjects[0]
            facetoken = faceobject["face_token"]
            newurlanalyse = faceplusplusurlanalyse + "&face_tokens=" + facetoken
            responseanalyse = requests.post(newurlanalyse)
            if responseanalyse.status_code != 200:
                # print(row['screen_name'] + ' Response error: ' + responseanalyse.status_code)
                return
            responsejsonanalyse = responseanalyse.json()
            if len(responsejsonanalyse["faces"]) == 1:
                facesanalyse = responsejsonanalyse["faces"]
                gender = facesanalyse[0]["attributes"]["gender"]["value"]
                gender = gender.lower()
                genderint = -1
                if gender == "female":
                    genderint = 1
                elif gender == "male":
                    genderint = 2

                age = facesanalyse[0]["attributes"]["age"]["value"]
                ethnicity = facesanalyse[0]["attributes"]["ethnicity"]["value"]
                db.user.update({"_id": row["_id"]}, {"$set": {"t_gender": genderint, "t_age": age, "t_eth": ethnicity, "t_f_api": 4}})
                # print(row['screen_name'] + ' ' + str(gender) + ' ' + str(age) + ' ' + str(ethnicity))
        else:
            db.user.update({"_id": row["_id"]}, {"$set": {"t_f_api": 1}})
            return

    except Exception as exception:
        print(row['screen_name'] + ' exception: ', exception)


def main():
    counter_general = 0
    try:
        # for any user which has never been analyzed
        users = db.user.find({"t_f_api": {"$exists": False}})
        print('Processing ' + str(users.count()) + ' missing users:')
        for res in users:
            try:
                counter_general = counter_general + 1

                if not isTweetEligibleForFaceOperation(res):
                    continue

                updateMongoWithImageDemographics(res)
                if divmod(counter_general, 10)[1] == 0:
                    print('Done ' + str(counter_general) + ' / ' + str(users.count()))

            except Exception as exception:
                print('Oops!  An error occurred in loop.  Try again... line: ' + str(res), exception)

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)


if __name__ == "__main__":
    main()

