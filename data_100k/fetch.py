import pandas as pd



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

    user_data = pd.read_csv('u.user', header=None)
    job_list = pd.read_csv('u.occupation', header=None)
    array = []
    for iteration in range(len(user_data)):
        current_user_job = user_data.iloc[iteration].values.item().split('|')[3]
        embedding = list(map(lambda x: 1 if x==current_user_job else 0, job_list))
        array.append(embedding)
    user_feat_indicators = np.array(array)

    user_data = pd.read_csv('u.user')
    job_list = pd.read_csv('u.occupation', header=None)
    job_list = []
    for index, row in user_data.iterrows():
        value = user_data.iloc[index].values.item()
        occ = value.split('|')[3]
        if occ not in job_list:
            job_list.append(occ)
    array = []
    for iteration in range(len(user_data)):
        current_user_job = user_data.iloc[iteration].values.item().split('|')[3]
        embedding = list(map(lambda x: 1 if x == current_user_job else 0, job_list))
        array.append(embedding)
    user_feat_indicators = np.array(array)

    return interaction_matrix, user_feat_indicators, item_feat_indicators
