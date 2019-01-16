# NewsAnalyzer

## Intro

## Installation Guide

#### Requirements
- Python (>3.4.0) and pip
- MongoDB
- Twitter API keys
- Python libs in requirements.txt
- (optional) Face++ keys

### Setting up the application

Clone the repository:

`git clone https://github.com/DataSciencePolimi/NewsAnalyzer.git`

Inside the SKE folder initialize a python environment

`virtualenv ske-env`

Activate it

`source ske-env/bin/activate`

The install the requirements

`pip install -r requirements.txt`

### Setting up keystore

Open `credential.json` and fill the values with your keys:

```
{
    "consumer_key" : "<twitter API consumer key>",
    "consumer_secret" : "<twitter API consumer secret>",
    "access_token" : "<twitter API access token>",
    "access_token_secret" : "<twitter API access token secret>",
    "faceplus_key" : "<face++ key (optional)>",
    "faceplus_secret" : "<face++ secret (optional)>"
}
```

### Setting up database

1. Download the NewsAnalyzer dataset from `https://doi.org/10.7910/DVN/5XRZLH` and unzip
2. Run command `mongod` to start a MongoDB server on localhost
3. Run `setup.py` script inside application folder




