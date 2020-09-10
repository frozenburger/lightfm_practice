'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.placeholder(tf.float32)
add_five = x + 5

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

sess.run(add_five, feed_dict = {x: [1, 3, 5]})
'''
'''
import tensorflow as tf

x = tf.Variable(tf.constant(10.0), dtype=tf.float32)
with tf.GradientTape() as tape:
    loss = tf.multiply(x, x)
    grad = tape.gradient(loss, x)

optimizer = tf.keras.optimizers.Adagrad(learning_rate=1.0)
for i in range(100):
    optimizer.minimize(loss, var_list=[x])
    #grad = optimizer.get_gradients(objective_function(), [x])
    #print(x, grad)
    gradient = tape.gradient(loss, x)
    optimizer.apply_gradients(gradient, x)
    print(x)
'''

import tensorflow as tf


x = tf.Variable(tf.constant(10.0), dtype=tf.float32)
opt = tf.keras.optimizers.SGD(learning_rate=0.05)


for i in range(10):
    with tf.GradientTape() as tape:
        def loss_func():
            return tf.multiply(x, x)
        grad = tape.gradient(loss_func(), x)
        opt.apply_gradients([(grad, x)])
        print('value : {}, gradient = {}, step = {}'.format(x.value(), grad, grad*0.05))




