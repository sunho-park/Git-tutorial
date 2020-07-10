import tensorflow as tf
import numpy as np

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]



# x (8, 4) // y (8, 3)
x = tf.placeholder('float', shape=[None, 4])
y = tf.placeholder('float', shape=[None, 3])

w = tf.Variable(tf.random_normal([4, 3]), name='weight') # wx + b
b = tf.Variable(tf.random_normal([1, 3]), name='bias') # [1], [3], [1, 3] 다됨  // 덧셈이라서 [3, 1]은 안됨

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1)) # categorical_crossentrophy


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss],
                                    feed_dict = {x:x_data, y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

# 최적의 w 와 b가 구해져있다.

    # predict

    a = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
    print(a, sess.run(tf.arg_max(a, 1))) 

    b = sess.run(hypothesis, feed_dict={x:[[1, 3, 4, 3]]})
    print(b, sess.run(tf.arg_max(b, 1))) 

    c = sess.run(hypothesis, feed_dict={x:[[11, 13, 4, 13]]})
    print(c, sess.run(tf.arg_max(c, 1))) # 이행에서 가장큰 값을 선정해라


    
    all = sess.run(hypothesis, feed_dict={x: [[1, 11, 7, 9], [1, 3, 4, 3], [11, 13, 4, 13]]})
    print(all, sess.run(tf.arg_max(all, 1)))

    # # # # # 
    # all = sess.run(hypothesis, feed_dict={x: [np.append(a, 0), np.append(b, 0), np.append(c, 0)]})
    # print(all, sess.run(tf.arg_max(all, 1)))







