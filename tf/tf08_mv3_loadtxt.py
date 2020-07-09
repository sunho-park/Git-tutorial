# mv : multi variable
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)


dataset = np.loadtxt('./data/csv/data-01-test-score.csv',
                        delimiter=',', dtype = np.float32)

x_data = dataset[:, 0:-1]
y_data = dataset[:, [-1]]

print(x_data.shape)     # (25, 3)
print(y_data.shape)     # (25, 1)

##################################################################################
x = tf.placeholder(tf.float32, shape=[None, 3]) 
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b # wx + b  matmul 행렬 곱 # 이자체가 activation = defalut  // linear

cost = tf.reduce_mean(tf.square(hypothesis-y)) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005) 
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1001):
    cost_val, hy_val, _= sess.run(
        [cost, hypothesis, train], feed_dict={x: x_data, y: y_data})
    
    if step % 10==0:
        print(step, "cost:", cost_val, "\n 예측값 : ", hy_val)

# https://hunkim.github.io/ml/







