import pandas as pd
import numpy as np

def movielens():
    train = pd.read_csv('data_100k/u1.base', sep='\t', header=None).drop(columns=3)
    test = pd.read_csv('data_100k/u1.test', sep='\t', header=None).drop(columns=3)
    train_keys = train.pivot(index=0, columns=1, values=2).keys()
    test_keys = test.pivot(index=0, columns=1, values=2).keys()
    test_index = test.pivot(index=0, columns=1, values=2).index

    train_added_keys = []
    test_added_keys = []
    test_added_index = []
    for i in np.arange(1682) + 1:
        if i not in train_keys:
            train.loc[len(train)] = [1, i, -1]
            train_added_keys.append(i)
        if i not in test_keys:
            test.loc[len(test)] = [1, i, -1]
            test_added_keys.append(i)
    for i in np.arange(943) + 1:
        if i not in test_index:
            test.loc[len(test)] = [i, 1, -1]
            test_added_index.append(i)

    train = train.pivot(index=0, columns=1, values=2)
    test = test.pivot(index=0, columns=1, values=2)

    train.values[(0 <= train.values) & (train.values <= 3)] = 0
    train.values[(4 <= train.values) & (train.values <= 5)] = 1
    test.values[(0 <= test.values) & (test.values <= 3)] = 0
    test.values[(4 <= test.values) & (test.values <= 5)] = 1
    train = train.fillna(-1).to_numpy()
    test = test.fillna(-1).to_numpy()
    # train : (0 * 35860) + (1 * 44140)
    # test : (0 * 8765) + (1 * 11235)

    user_data = pd.read_csv('data_100k/u.user', header=None)
    job_list = pd.read_csv('data_100k/u.occupation', header=None)
    array = []
    for iteration in range(len(user_data)):
        current_user_job = user_data.iloc[iteration].values.item().split('|')[3]
        job_index = list(map(lambda x: 1 if x == current_user_job else 0, job_list[0]))
        sex_index = [1, 0] if user_data.iloc[iteration].values.item().split('|')[2] == 'M' else [0, 1]
        array.append(job_index + sex_index)
    user_feat_indicators = np.array(array)

    movies_data = pd.read_csv('data_100k/u.item', sep='|', encoding='latin-1', header=None)
    genre_df = pd.read_csv('data_100k/u.genre', header=None)[0]
    array = []
    for iteration in range(len(movies_data)):
        # genre index list (0s and 1s) * 19
        current_movie_genres = movies_data.loc[iteration][5:].to_list()
        array.append(current_movie_genres)
    item_feat_indicators = np.array(array)

    movies_dict = movies_data[1].to_dict()

    return train, test, user_feat_indicators, item_feat_indicators, movies_dict
