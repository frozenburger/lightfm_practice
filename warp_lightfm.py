import tensorflow as tf
import numpy as np

class model():
    def __init__(self, interactions, test_data, user_feat_indicators, item_feat_indicators, labels, k_embeddings=10):

        def array_to_pairs(arr):
            positives = {}
            negatives = {}
            for i in range(arr.shape[0]):
                pos_temp = []
                neg_temp = []
                for j in range(arr.shape[1]):
                    if arr[i][j] == 1:
                        pos_temp.append(j)
                    elif arr[i][j] == 0:
                        neg_temp.append(j)
                positives[i] = pos_temp
                negatives[i] = neg_temp
            return positives, negatives

        init_xav = tf.initializers.GlorotUniform()
        self.labels = labels
        self.test_data = test_data

        self.size_u, self.size_i = interactions.shape
        self.size_uf, self.size_if = user_feat_indicators.shape[1], item_feat_indicators.shape[1]
        self.k_embeddings = k_embeddings

        self.interactions = tf.constant(interactions, dtype=tf.float32) #유저x상품 상호작용이 positive 시 0, negative 시 1
        self.positive_pairs, self.negative_pairs = array_to_pairs(interactions)

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


    '''
    무작위 유저 선택
    그 유저의 positive interaction 1개 무작위 선택
    그 유저의 negative interaction 1개 무작위 선택
    만약 negative interaction의 score가 positive보다 높으면 틀린 것으로 간주
    두 score를 비교 후 옳게 되어 있으면 패스, 틀렸을 경우 두 score의 차이를 loss로 반환
    '''
    def warp_loss(self):
        random_user_index = np.random.randint(0, self.size_u)
        random_pos_item_index = np.random.randint(0, len(self.positive_pairs[random_user_index]))
        random_neg_item_index = np.random.randint(0, len(self.negative_pairs[random_user_index]))

        scores = self.get_prediction_matrix()
        pos_score = scores[random_user_index][random_pos_item_index]
        neg_score = scores[random_user_index][random_neg_item_index]
        if pos_score > neg_score :
            pass
        elif pos_score <= neg_score :
            return neg_score - pos_score



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
        
    def linear_loss(self):
        positive_scores = tf.multiply(self.get_prediction_matrix(), self.positives)
        negative_scores = self.negatives - tf.multiply(self.get_prediction_matrix(), self.negatives)
        squared_diff = tf.math.squared_difference(self.true_matrix, positive_scores+negative_scores)
        return tf.sqrt(tf.reduce_sum(squared_diff) / tf.reduce_sum(self.positives + self.negatives))
    '''



'''
# EXECUTION SHORTCUT
from data_100k.fetch import movielens
from tf2_lightfm import model
d1, d2, d3, d4, d5 = movielens()
cls = model(d1, d2, d3, d4, d5)
cls.train(30)
'''