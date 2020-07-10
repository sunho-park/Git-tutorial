# import tensorflow as tf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_train.shape)        # (60000,)
print(y_test.shape)         # (10000,)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

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
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

x_img = tf.reshape(x, [-1, 28, 28, 1])

# b1 = tf.Variable(tf.random_normal([100]), name="bias")
w1 = tf.get_variable("w1", shape=[3, 3, 1, 32])      
# Conv2D(32, (3, 3), input_shape=(28, 28, 1)) 32 output 
print("================================================")
print(w1)                   # shape=(3, 3, 1, 32)
print("================================================")

L1 = tf.nn.conv2d(x_img, w1, strides=[1, 1, 1, 1], padding='SAME')
print("================================================")
print(L1)                   # shape=(?, 28, 28, 32)
print("================================================")

L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # ksize kernel size / ksize stride 가운데만먹힘 2, 2 
print(L1)                     # shape=(?, 14, 14, 32)
#################################################################################################
w2 = tf.get_variable("w2", shape=[3, 3, 32, 64])      
# Conv2D(32, (3, 3), in=(28, 28, 1)) 32 output 
L2 = tf.nn.conv2d(L1, w2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # ksize kernel size / ksize stride 가운데만먹힘 2, 2 
print(L2)                     # shape=(?, 7, 7, 64)  maxpool때문에
#################################################################################################
L2_flat = tf.reshape(L2, [-1, 7*7*64])
w3 = tf.get_variable("w3", shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer()) #커널초기화
b3 = tf.Variable(tf.random_normal([10]))
print(L2_flat)
# Conv2D에는 bias 계산이 자동으로 되기 때문에 b 따로 명시 안해줘도 됨

hypothesis = tf.nn.softmax(tf.matmul(L2_flat, w3) + b3)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

    
# Training cycle
for epoch in range(training_epochs):    # training_epochs : 15
    avg_cost = 0
    for i in range(total_batch):        # total_batch : 600
############이 부분 구현할 것#####################################
        start = i * batch_size    # 0
        end = start + batch_size  # 100           

        batch_xs, batch_ys = x_train[start:end], y_train[start:end]        
#################################################################        
        feed_dict = {x:batch_xs, y:batch_ys, keep_prob:0.7}
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c/total_batch
    
    print('Epoch : ', '%04d' %(epoch+1), 'cost = ', '{:.9f}'.format(avg_cost))

print('훈련 끝')

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test, keep_prob:1}))

