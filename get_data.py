# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:25:07 2020

@author: Adi
"""
import requests
import json, sys
import pandas as pd

API_KEY = sys.argv[1]
cachefile = 'omdb-cache.dbm'
NOT_FOUND = 'notfound'
request_limit = False

def get_omdb_data(imdb_id):
    
    global request_limit
    if request_limit:
        return None
    if not imdb_id.startswith('tt'):
        raise ValueError('movies only')

    url = 'http://www.omdbapi.com/?i=%s&apikey=%s&plot=full' % (imdb_id, API_KEY)
    print('fetching', url)
    r = requests.get(url)

    data = json.loads(r.text)
    if data['Response'] == 'False':
        if data['Error'] == 'Error getting data.':
            return NOT_FOUND
        elif data['Error'] == 'Request limit reached!':
            print("Request limit reached")
            request_limit = True
            return None
        else:
            raise ValueError(data['Error'])

    return data

def main():
    infile = './wikidata-movies.json.gz'
    movie_data = pd.read_json(infile, orient='records', lines=True)
    print(movie_data)

    test_db = movie_data['imdb_id'].apply(get_omdb_data)
    test_db = test_db[test_db.notnull()]
    print(test_db)
    
    test_db.to_json('./omdb-data.json.gz', orient='records', lines=True, compression='gzip')


if __name__ == '__main__':
    main()
