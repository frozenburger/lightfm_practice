import tensorflow as tf
import numpy as np

class model():
    def __init__(self, interactions, user_feat_indicators, item_feat_indicators, k_embeddings=30):

        init_xav = tf.initializers.GlorotUniform()

        self.size_u, self.size_i = interactions.shape
        self.size_uf, self.size_if = user_feat_indicators.shape[1], item_feat_indicators.shape[1]
        self.k_embeddings = k_embeddings

        self.interactions = tf.constant(interactions, dtype=tf.float32) #유저x상품 상호작용이 positive 시 0, negative 시 1
        self.user_feat_indicators = tf.Variable(user_feat_indicators, dtype=tf.float32) #유저피쳐 해당시 1, 아닐시 0 ??
        self.item_feat_indicators = tf.Variable(item_feat_indicators, dtype=tf.float32) #상품피쳐 해당시 1, 아닐시 0
        self.user_feat_embeddings = tf.Variable(init_xav(shape=(self.size_uf, self.k_embeddings)), dtype=tf.float32) #학습시킬 각 유저피쳐 임베딩
        self.item_feat_embeddings = tf.Variable(init_xav(shape=(self.size_if, self.k_embeddings)), dtype=tf.float32) #학습시킬 각 상품피쳐 임베딩
        self.user_bias_vector = tf.Variable(init_xav(shape=(self.size_u,)), dtype=tf.float32) # ??
        self.item_bias_vector = tf.Variable(init_xav(shape=(self.size_i,)), dtype=tf.float32) # ??


    def get_prediction_matrix(self):
        return tf.sigmoid(
            tf.matmul(
                tf.matmul(self.user_feat_indicators, self.user_feat_embeddings),
                tf.transpose(tf.matmul(self.item_feat_indicators, self.item_feat_embeddings))
            ) + tf.transpose([self.user_bias_vector]) + self.item_bias_vector
        )

    def objective_function(self):
        return (-1) * tf.reduce_prod(tf.abs(self.interactions - self.get_prediction_matrix()))

    def train(self, epoch=30):
        opt = tf.keras.optimizers.Adagrad(learning_rate=0.05)
        for i in range(epoch):
            print('epoch : {}, loss : {}'.format(i, self.objective_function()))
            opt.minimize(self.objective_function, var_list=[self.user_feat_embeddings, self.item_feat_embeddings])
