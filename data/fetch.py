# preprocessing
import pandas as pd
import numpy as np

def movielens():
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')

    genre_list = []
    for index, row in movies.iterrows():
        for item in row.genres.split('|'):
            if item not in genre_list:
                genre_list.append(item)

    array = []
    for iteration in range(len(movies)):
        current_movie_genres = movies.loc[iteration].genres.split('|')
        embedding = list(map(lambda x : 1 if x in current_movie_genres else 0, genre_list))
        array.append(embedding)
    item_feat_indicators = np.array(array)

    ratings = ratings.drop(columns = ['timestamp'])
    missing_list = []
    movie_lst = ratings.movieId.unique()
    for id in movies.movieId:
        if int(id) not in movie_lst:
            missing_list.append(id)
    for id in missing_list:
        ratings = ratings.append({'userId':1, 'movieId':int(id), 'rating':0.0}, ignore_index=True)
    ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    ratings.values[4 <= ratings.values] = 0
    ratings.values[ratings.values <= 3] = 1
    interaction_matrix = ratings

    rng = np.random
    lst = []
    for i in range(interaction_matrix.shape[0]):
        num = rng.randint(4)
        row = [1] * num + [0] * (20-num)
        rng.shuffle(row)
        lst.append(row)
    user_feat_indicators = np.array(lst)

    return interaction_matrix, user_feat_indicators, item_feat_indicators
