'''
import tensorflow as tf

x1 = tf.Variable(10.0)
x2 = tf.Variable(10.0)

def obj_func(x1, x2):
    return tf.pow(x1, 2) - tf.multiply(x1, 3) + tf.pow(x2, 2)
def minimize_function():
    return tf.pow(x1, 2) - tf.multiply(x1, 3) + tf.pow(x2, 2)


opt = tf.keras.optimizers.SGD(learning_rate=0.1)
for i in range(50):
    print ('y = {:.1f}, x1 = {:.1f}, x2 = {:.1f}'.format(obj_func(x1, x2).numpy(), x1.numpy(), x2.numpy()))
    opt.minimize(minimize_function, var_list=[x1, x2])


현재 x1, x2의 값을 이용해 objective function의 값을 계산하고 출력한다.
minimize function의 값이 최소가 되도록 var_list에 들어있는 tf.variable들을 조정한다.
'''
'''
import tensorflow as tf
import numpy as np
rng = np.random
size_u, size_i, size_f = 3, 5, 4

user_embeddings_matrix = tf.Variable(np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]]), dtype=tf.float32)
item_embeddings_matrix = tf.Variable(np.array([[0, 0, 0, 1, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]]), dtype=tf.float32)
user_bias_vector = tf.Variable(np.array([0, 1, 0]), dtype=tf.float32)
item_bias_vector = tf.Variable(np.array([0, 0, 0, 0, 1]), dtype=tf.float32)
interaction_matrix = tf.constant(np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 1, 0, 0]]))

def get_pred_matrix(matrix1, matrix2, bias_vector1, bias_vector2):
    pred_matrix = tf.sigmoid(tf.matmul(matrix1, matrix2) + tf.transpose([bias_vector1]) + [bias_vector2])
    return pred_matrix
def objective_function():
    return tf.reduce_prod(tf.abs(interaction_matrix - tf.sigmoid(tf.matmul(user_embeddings_matrix, item_embeddings_matrix) + tf.transpose([user_bias_vector]) + [item_bias_vector])))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
for i in range(5):
    print('Calculated Likelyhood : ', tf.reduce_prod(get_pred_matrix(user_embeddings_matrix, item_embeddings_matrix, user_bias_vector, item_bias_vector)))
    optimizer.minimize(objective_function, var_list=[user_embeddings_matrix, item_embeddings_matrix, user_bias_vector, item_bias_vector])
'''


# preprocessing
import pandas as pd
import numpy as np
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

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
TFVAR_item_embeddings = np.transpose(np.array(array))

ratings = ratings.drop(columns = ['timestamp'])
missing_list = []
movie_lst = ratings.movieId.unique()
for id in movies.movieId:
    if int(id) not in movie_lst:
        missing_list.append(id)
for id in missing_list:
    ratings = ratings.append({'userId':1, 'movieId':int(id), 'rating':0.0}, ignore_index=True)
ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
ratings.values[4 <= ratings.values] = 1
ratings.values[ratings.values <= 3] = 0
TFVAR_interaction_matrix = ratings





import tensorflow as tf
import numpy as np
rng = np.random
size_u, size_i, size_f = 671, 9125, 20

user_embeddings_matrix = tf.Variable(rng.rand(size_u, size_f), dtype=tf.float32)
item_embeddings_matrix = tf.Variable(TFVAR_item_embeddings, dtype=tf.float32)
user_bias_vector = tf.Variable(rng.rand(size_u), dtype=tf.float32)
item_bias_vector = tf.Variable(rng.rand(size_i), dtype=tf.float32)
interaction_matrix = tf.constant(TFVAR_interaction_matrix, dtype=tf.float32)

def get_pred_matrix(matrix1, matrix2, bias_vector1, bias_vector2):
    pred_matrix = tf.sigmoid(tf.matmul(matrix1, matrix2) + tf.transpose([bias_vector1]) + [bias_vector2])
    return pred_matrix
def objective_functions():
    return tf.reduce_prod(tf.abs(interaction_matrix - tf.sigmoid(tf.matmul(user_embeddings_matrix, item_embeddings_matrix) + tf.transpose([user_bias_vector]) + [item_bias_vector])))

optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.5)
for i in range(30):
    print('Calculated Likelyhood : ', tf.reduce_prod(get_pred_matrix(user_embeddings_matrix, item_embeddings_matrix, user_bias_vector, item_bias_vector)))
    optimizer.minimize(objective_function, var_list=[user_embeddings_matrix, item_embeddings_matrix, user_bias_vector, item_bias_vector])
