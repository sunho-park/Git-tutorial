# import tensorflow as tf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_train.shape)        # (60000,)
print(y_test.shape)         # (10000,)

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

print(x_train.shape)        # (60000, 784)
print(x_test.shape)         # (10000, 784)

# 원핫 인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


print(x_train.shape, x_test.shape)      # (60000, 784) (10000, 784)
print(y_train.shape, y_test.shape)      # (60000, 10) (10000, 10)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/ batch_size)     # 60000 / 100

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)           # dropout

# b1 = tf.Variable(tf.random_normal([100]), name="bias")
w1 = tf.get_variable("w1", shape=[784, 512], 
                        initializer=tf.contrib.layers.xavier_initializer()) #커널초기화
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.selu(tf.matmul(x, w1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

print('w1', w1)      # <tf.Variable 'w1:0' shape=(784, 512) dtype=float32_ref>
print('b1', b1)      # <tf.Variable 'Variable:0' shape=(512,) dtype=float32_ref>
print('L1', L1)      # Tensor("Selu:0", shape=(?, 512), dtype=float32)
print('L1', L1)      # Tensor("dropout/mul_1:0", shape=(?, 512), dtype=float32)


# LAYER
w2 = tf.get_variable("w2", shape=[512, 512], 
                        initializer=tf.contrib.layers.xavier_initializer()) #커널초기화

b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# LAYER
w3 = tf.get_variable("w3", shape=[512, 512], 
                        initializer=tf.contrib.layers.xavier_initializer()) #커널초기화

b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.selu(tf.matmul(L2, w3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# LAYER
w4 = tf.get_variable("w4", shape=[512, 256], 
                        initializer=tf.contrib.layers.xavier_initializer()) #커널초기화

b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.selu(tf.matmul(L3, w4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# LAYER
w5 = tf.get_variable("w5", shape=[256, 10], 
                        initializer=tf.contrib.layers.xavier_initializer()) #커널초기화

b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(L4, w5) + b5)


cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    # Training cycle
    for epoch in range(training_epochs):    # training_epochs : 15
        avg_cost = 0
        start = 0
        for i in range(total_batch):        # total_batch : 600
    ############이 부분 구현할 것#####################################
            order=i*batch_size
            batch_xs, batch_ys = x_train[order:+order+batch_size], y_train[order:order+batch_size]
    #################################################################
            start = i * batch_size    # 0
            end = start + batch_size  # 100           

            batch_xs, batch_ys = x_train[start:end], y_train[start:end]        
    #################################################################        
            feed_dict = {x:batch_xs, y:batch_ys, keep_prob:0.7}
            c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
            avg_cost += c/total_batch
        
        print('Epoch : ', '%04d' %(epoch+1), 'cost = ', '{:.9f}'.format(avg_cost))

    print('훈련 끝')

   
    print('Acc : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test, keep_prob:1}))
