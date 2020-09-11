import tensorflow as tf
import numpy as np

class model():
    def __init__(self, interactions, test_data, user_feat_indicators, item_feat_indicators, labels, k_embeddings=10):

        init_xav = tf.initializers.GlorotUniform()
        self.labels = labels
        self.test_data = test_data

        self.size_u, self.size_i = interactions.shape
        self.size_uf, self.size_if = user_feat_indicators.shape[1], item_feat_indicators.shape[1]
        self.k_embeddings = k_embeddings

        self.interactions = tf.constant(interactions, dtype=tf.float32) #유저x상품 상호작용이 positive 시 0, negative 시 1
        self.positives = tf.constant((interactions == 1).astype(float), dtype=tf.float32)
        self.negatives = tf.constant((interactions == 0).astype(float), dtype=tf.float32)
        self.true_matrix = self.positives + self.negatives

        self.user_feat_indicators = tf.constant(user_feat_indicators, dtype=tf.float32) #유저피쳐 해당시 1, 아닐시 0 ??
        self.item_feat_indicators = tf.constant(item_feat_indicators, dtype=tf.float32) #상품피쳐 해당시 1, 아닐시 0
        self.user_feat_embeddings = tf.Variable(init_xav(shape=(self.size_uf, self.k_embeddings)), dtype=tf.float32) #학습시킬 각 유저피쳐 임베딩
        self.item_feat_embeddings = tf.Variable(init_xav(shape=(self.size_if, self.k_embeddings)), dtype=tf.float32) #학습시킬 각 상품피쳐 임베딩
        self.user_feat_bias_vector = tf.Variable(init_xav(shape=(self.size_uf,)), dtype=tf.float32) # ??
        self.item_feat_bias_vector = tf.Variable(init_xav(shape=(self.size_if,)), dtype=tf.float32) # ??

        self.opt = tf.keras.optimizers.Adagrad(learning_rate=0.1)

    def get_prediction_matrix(self):
        return tf.sigmoid(
            tf.matmul(
                tf.matmul(self.user_feat_indicators, self.user_feat_embeddings),
                tf.transpose(tf.matmul(self.item_feat_indicators, self.item_feat_embeddings))
            )
            + tf.transpose([tf.linalg.matvec(self.user_feat_indicators, self.user_feat_bias_vector)])
            + tf.linalg.matvec(self.item_feat_indicators, self.item_feat_bias_vector)
        )

    def linear_loss(self):
        positive_scores = tf.multiply(self.get_prediction_matrix(), self.positives)
        negative_scores = self.negatives - tf.multiply(self.get_prediction_matrix(), self.negatives)
        squared_diff = tf.math.squared_difference(self.true_matrix, positive_scores+negative_scores)
        return tf.sqrt(tf.reduce_sum(squared_diff) / tf.reduce_sum(self.positives + self.negatives))

    def train(self, epoch=30):
        for i in range(epoch):
            with tf.GradientTape() as tape:
                gradients = tape.gradient(self.linear_loss(), [self.user_feat_embeddings, self.item_feat_embeddings, self.user_feat_bias_vector, self.item_feat_bias_vector])
                self.opt.apply_gradients(zip(gradients, [self.user_feat_embeddings, self.item_feat_embeddings, self.user_feat_bias_vector, self.item_feat_bias_vector]))
                if i%10 == 9:
                    self.validate()
                    print('epoch : {}, loss : {}, learning rate : {}'.format(i+1, self.linear_loss().numpy(), self.opt.get_config()['learning_rate']))
                    print('-------------------------------------------------------------')
                #print(list(map(lambda x : np.reciprocal(x.numpy()).astype(float), gradients))) # (21, 10) / (19, 10) / (21, ) / (19, )

    def get_optimizer(self):
        return self.opt

    def get_prediction_hitrate(self, user_id):
        user_pred = self.get_prediction_matrix().__array__()[user_id]
        user_history = self.interactions.__array__()[user_id]
        mask_watched = (user_history != -1).astype(int)
        watched_scores = np.multiply(mask_watched, user_pred).argsort()[-20:]
        mask_unwatched = (user_history == -1).astype(int)
        unwatched_scores = np.multiply(mask_unwatched, user_pred).argsort()[-20:]

        count = 0
        for movie_id in unwatched_scores:
            if self.test_data[user_id][movie_id] != -1 :
                count += 1
        return 100*(count/20)

    def predict(self, user_id):
        user_pred = self.get_prediction_matrix().__array__()[user_id]
        user_history = self.interactions.__array__()[user_id]
        mask_watched = (user_history != -1).astype(int)
        watched_scores = np.multiply(mask_watched, user_pred).argsort()[-20:]
        mask_unwatched = (user_history == -1).astype(int)
        unwatched_scores = np.multiply(mask_unwatched, user_pred).argsort()[-20:]

        print('Positives : ')
        for movie_id in watched_scores:
            print('movie : {}, score : {}'.format(self.labels[movie_id], user_pred[movie_id]))
        print('\nPredictions : ')
        for movie_id in unwatched_scores:
            print('movie : {}, score : {}, actual value : {}'.format(self.labels[movie_id], user_pred[movie_id], self.test_data[user_id][movie_id]))

    def validate(self):
        global_mean = np.mean(list(map(lambda x : self.get_prediction_hitrate(x), np.arange(943))))
        print(global_mean, '% hit rate overall on 943 users.')


    '''
    # DEPRECATED OPERATIONS
    def train(self, epoch=30):
        opt = tf.keras.optimizers.Adagrad(learning_rate=0.05)
        for i in range(epoch):
            print('epoch : {}, loss : {}'.format(i, self.rms_loss()))
            opt.minimize(self.rms_loss, var_list=[self.user_feat_embeddings, self.item_feat_embeddings, self.user_feat_bias_vector, self.item_feat_bias_vector])
    def objective_function(self):
        mask = tf.math.is_nan(self.interactions)
        scores_with_nan = self.interactions - self.get_prediction_matrix()
        scores_without_nan = tf.where(mask, tf.ones_like(self.interactions), scores_with_nan)
        return (-1) * tf.reduce_prod(tf.abs(scores_without_nan))

    def rms_loss(self):
        mask = (tf.nn.relu(self.interactions) != self.interactions)
        squared_diff = tf.math.squared_difference(self.interactions, self.get_prediction_matrix())
        squared_diff_relevant = tf.where(mask, tf.zeros_like(squared_diff), squared_diff)
        return tf.sqrt(tf.reduce_mean(squared_diff_relevant))
    '''



'''
# EXECUTION SHORTCUT
from data_100k.fetch import movielens_split
from tf2_lightfm import model
d1, d2, d3, d4, d5 = movielens()
cls = model(d1, d2, d3, d4, d5)
cls.train(30)
'''