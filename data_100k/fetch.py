# preprocessing
import pandas as pd
import numpy as np
from lightfm.datasets.movielens import fetch_movielens

def movielens(to_binary=False):
    ratings_data = pd.read_csv('data_100k/u.data', sep='\t', header=None).drop(columns=3)
    ratings_data = ratings_data.pivot(index=0, columns=1, values=2)
    if to_binary==True:
        ratings_data.values[ratings_data.values <= 3] = 0
        ratings_data.values[4 <= ratings_data.values] = 1
    ratings_data = ratings_data.fillna(-1)
    interaction_matrix = ratings_data.to_numpy()

    user_data = pd.read_csv('data_100k/u.user', header=None)
    job_list = pd.read_csv('data_100k/u.occupation', header=None)
    array = []
    for iteration in range(len(user_data)):
        current_user_job = user_data.iloc[iteration].values.item().split('|')[3]
        embedding = list(map(lambda x: 1 if x == current_user_job else 0, job_list[0]))
        array.append(embedding)
    user_feat_indicators = np.array(array)

    movies_data = pd.read_csv('data_100k/u.item', sep='|', encoding='latin-1', header=None)
    genre_df = pd.read_csv('data_100k/u.genre', header=None)[0]
    genre_list = list(map(lambda x: x.split('|')[1], genre_df.to_list()))
    array = []
    for iteration in range(len(movies_data)):
        current_movie_genres = movies_data.loc[iteration][5:].to_list()
        array.append(current_movie_genres)
    item_feat_indicators = np.array(array)

    movies_dict = movies_data[1].to_dict()

    return interaction_matrix, user_feat_indicators, item_feat_indicators, movies_dict

def movielens_split():
    data = fetch_movielens()
    ratings_data = data['train'].toarray()
    ratings_data[ratings_data==0] = -1
    ratings_data[(0 <= ratings_data) & (ratings_data <= 3)] = 0
    ratings_data[(4 <= ratings_data) & (ratings_data <= 5)] = 1
    interaction_matrix = ratings_data

    user_data = pd.read_csv('data_100k/u.user', header=None)
    job_list = pd.read_csv('data_100k/u.occupation', header=None)
    array = []
    for iteration in range(len(user_data)):
        current_user_job = user_data.iloc[iteration].values.item().split('|')[3]
        embedding = list(map(lambda x: 1 if x == current_user_job else 0, job_list[0]))
        array.append(embedding)
    user_feat_indicators = np.array(array)

    movies_data = pd.read_csv('data_100k/u.item', sep='|', encoding='latin-1', header=None)
    genre_df = pd.read_csv('data_100k/u.genre', header=None)[0]
    genre_list = list(map(lambda x: x.split('|')[1], genre_df.to_list()))
    array = []
    for iteration in range(len(movies_data)):
        current_movie_genres = movies_data.loc[iteration][5:].to_list()
        array.append(current_movie_genres)
    item_feat_indicators = np.array(array)

    movies_dict = movies_data[1].to_dict()

    test_data = data['test'].toarray()

    return interaction_matrix, user_feat_indicators, item_feat_indicators, movies_dict, test_data