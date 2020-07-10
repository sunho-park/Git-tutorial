# 케라스는 2.3.1 로 설치
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_train.shape)        # (60000,)
print(y_test.shape)         # (10000,)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print(x_train.shape)        # (60000, 784)
print(x_test.shape)         # (10000, 784)

# 원핫 인코딩
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))
y_train=y_train.reshape(-1,10)
y_test=y_test.reshape(-1,10)


x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
# layer
w1 = tf.Variable(tf.random_normal([784, 100]), name='weight1')
b1 = tf.Variable(tf.random_normal([100]), name='bias1')
dense1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# layer2
w2 = tf.Variable(tf.random_normal([100, 50]), name='weight2')
b2 = tf.Variable(tf.random_normal([50]), name='bias2')
dense2 = tf.sigmoid(tf.matmul(dense1, w2) + b2)

# layer3
w3 = tf.Variable(tf.random_normal([50, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
hypothesis = tf.nn.softmax(tf.matmul(dense2, w3) + b3)

# layer3
# w3 = tf.Variable(tf.zeros([784, 10]), name='weight3')
# b3 = tf.Variable(tf.zeros([10]), name='bias3')
# hypothesis = tf.nn.softmax(tf.matmul(x, w3) + b3)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(loss)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        _, cost_val = sess.run([optimizer, loss],
                                    feed_dict = {x:x_train, y:y_train})

        if step % 200 == 0:
            print(step, cost_val)
    h, c, a = sess.run([hypothesis, prediction, accuracy],
                                feed_dict={x:x_test, y:y_test})
    print("\n Hypothesis : ", h, "\n Correct (y) : ", 
            "\n Accuracy : ", a)
