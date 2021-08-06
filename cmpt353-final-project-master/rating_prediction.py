import json
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from textblob import TextBlob

pd.options.mode.chained_assignment = None  # default='warn'

def map_dict(words):
    d = dict()
    i = 0
    for word in words:
        if word not in d:
            d[word] = i
            i += 1
    return d

def clean_data(df):
    df = df[[ 'Year', 'Rated', 'Genre', 'Director', 'Writer', 'Actors', 'Language', 'Country', 'imdbRating', 'Production']]

    # clean the imdb rating
    df = df[ df['imdbRating'] != 'N/A' ]
    df['imdbRating'] = df['imdbRating'].astype('float32')
    df['imdbRating'] = df['imdbRating'].astype('int32')
    df = df[df['imdbRating'] >= 0] 
    df = df[df['imdbRating'] <= 10]

    # clean the year
    df['Year'] = df['Year'].astype(str).str[0:4].astype(int)

    # clean the Genre 
    df['Genre'] = df['Genre'].str.split(", ")
    df['Genre1'] = df['Genre'].map(lambda x: x[0])
    df['Genre2'] = df['Genre'].map(lambda x: x[1] if len(x) > 1 else 'N/A')
    df['Genre3'] = df['Genre'].map(lambda x: x[2] if len(x) > 2 else 'N/A')
    df = df.drop('Genre', axis=1)
    df = df.replace({"Genre1": map_dict(df['Genre1'].unique())})
    df = df.replace({"Genre2": map_dict(df['Genre2'].unique())})
    df = df.replace({"Genre3": map_dict(df['Genre3'].unique())})

    # clean the rated
    df['Rated'] = df['Rated'].replace(['Not Rated', 'NOT RATED', 'Unrated', 'UNRATED'], 'N/A')
    df = df.replace({"Rated": map_dict(df['Rated'].unique())})

    # clean the language
    df['Language'] = df['Language'].str.split(", ")
    df['Language1'] = df['Language'].map(lambda x: x[0])
    df['Language2'] = df['Language'].map(lambda x: x[1] if len(x) > 1 else 'N/A')
    df = df.drop('Language', axis=1)
    df = df.replace({"Language1": map_dict(df['Language1'].unique())})
    df = df.replace({"Language2": map_dict(df['Language2'].unique())})

    # clean the country
    df['Country'] = df['Country'].str.split(", ")
    df['Country1'] = df['Country'].map(lambda x: x[0])
    df['Country2'] = df['Country'].map(lambda x: x[1] if len(x) > 1 else 'N/A')
    df = df.drop('Country', axis=1)
    df = df.replace({"Country1": map_dict(df['Country1'].unique())})
    df = df.replace({"Country2": map_dict(df['Country2'].unique())})

    print(df)
    return df

def get_XY(df):
    X = df[['Year', 'Genre1', 'Genre2', 'Genre3', 'Rated', 'Language1', 'Language2', 'Country1', 'Country2']].to_numpy()
    # X = np.stack([X], axis=1)
    y = df['imdbRating'].to_numpy()
    return X, y

def create_model():
    # model = GaussianNB()
    # model = KNeighborsClassifier(n_neighbors=75)
    model = RandomForestClassifier(n_estimators=50, max_depth=10)
    return model


def main(in_directory, out_directory):
    df = pd.read_json(in_directory, lines=True)

    # clean dataset
    df = clean_data(df)

    # get X,y and split data sets
    X, y = get_XY(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # fit and test the model
    model = create_model()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

if __name__ == "__main__":
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)