import plotly
import plotly.plotly as py
import pandas as pd
import json
import pymongo
import pprint as pp
import numpy as np

fileKeys = open('../credentials/credentialsTwitter.json').read()
keys = json.loads(fileKeys)
plotly.tools.set_credentials_file(username=keys['plotly_user'], api_key=keys['plotly_key'])

def plot_map(data, title, legend_title):
    """
    This function plots a series of values into US states map.
    Requires plotly to be set up with credentials on local machine.

    :parameter data
    must be an array of tuples in form [(<state name>, <value>), ...]
    state names being Alabama, Alaska, etc...

    :parameter title
    main title of the plot

    :parameter legend_title
    name of the value scale

    """

    # parse state codes from file
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
    for col in df.columns:
        df[col] = df[col].astype(str)

    # create state_name - code dictionary
    country_codes = {}
    for i in range(0, len(df['code'])):
        country_codes[df['state'][i]] = df['code'][i]

    # prepare data
    data_codes = []
    data_values = []
    data_texts = []
    for d in data:
        try:
            state = d[0].replace('State of ', '').replace('District of ', '')
            stateid = country_codes[state]
            statevalue = d[1]
            data_codes.append(stateid)
            data_values.append(statevalue)
            data_texts.append(state + ': ' + str(d[1]))
        except KeyError:
            print('Translation error for datum: ' + d[0])
            pass

    scl = [[0.0, 'rgb(242,240,247)'], [0.2, 'rgb(218,218,235)'], [0.4, 'rgb(188,189,220)'], [0.6, 'rgb(158,154,200)'], [0.8, 'rgb(117,107,177)'], [1.0, 'rgb(84,39,143)']]

    data = [dict(
        type='choropleth',
        # colorscale=scl,
        autocolorscale=True,
        locations=data_codes,
        z=data_values,
        locationmode='USA-states',
        text=data_texts,
        marker=dict(
            line=dict(
                color='rgb(255,255,255)',
                width=2
            )),
        colorbar=dict(
            title=legend_title)
    )]

    layout = dict(
        title=title,
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'),
    )

    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename='d3-cloropleth-map')


def plot_map_scatter(data, title, legend_title):
    lat = []
    long = []
    texts = []
    for d in data:
        coord = d['coordinates'].split(' ')
        lat.append(float(coord[1]))
        long.append(float(coord[0]))
        texts.append(d['name'])

    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')
    df.head()

    # df['text'] = df['airport'] + '' + df['city'] + ', ' + df['state'] + '' + 'Arrivals: ' + df['cnt'].astype(str)

    scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
        [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

    data = [ dict(
            type = 'scattergeo',
            locationmode = 'USA-states',
            lon = long,
            lat = lat,
            text = texts,
            mode = 'markers',
            marker = dict(
                size = 4,
                opacity = 0.4,
                reversescale = True,
                autocolorscale = False,
                symbol = 'circle',
                line = dict(
                    width=1,
                    color='rgba(102, 102, 102)'
                ),
                colorscale = scl,
                cmin=0,
                color=list(np.ones(len(texts))),
                cmax=1,
                colorbar=dict(
                    title="Incoming flightsFebruary 2011"
                )
            ))]

    layout = dict(
            title = title,
            colorbar = False,
            geo = dict(
                scope='usa',
                projection=dict( type='albers usa' ),
                showland = True,
                landcolor = "rgb(250, 250, 250)",
                subunitcolor = "rgb(217, 217, 217)",
                countrycolor = "rgb(217, 217, 217)",
                countrywidth = 2,
                subunitwidth = 2
            ),
        )

    fig = dict( data=data, layout=layout )
    py.iplot( fig, validate=False, filename='d3-airports' )


def plot_users_location():
    client = pymongo.MongoClient('localhost:27017')
    db = client.NewsAnalyzer

    states = {}
    for u in db.user.find({'y_geocode.address.country_code': 'US'}):
        for d in u['y_geocode']['address']['Components']:
            if d['kind'] == 'province':
                if d['name'] not in states:
                    states[d['name']] = 0
                states[d['name']] += 1

    states = sorted(states.items(), key=lambda x: x[1], reverse=True)
    pp.pprint(states)
    plot_map(states, 'Distribuzione geografica utenti US', 'Numero di utenti')


def plot_users_coordinates():
    client = pymongo.MongoClient('localhost:27017')
    db = client.NewsAnalyzer

    input_data = []
    for u in db.user.find({'y_geocode.address.country_code': 'US'}):
        input_data.append({'name': u['screen_name'], 'coordinates': u['y_geocode']['coordinates']})

    plot_map_scatter(input_data, 'Coordinate utenti US', None)


# plot_users_location()
plot_users_coordinates()



