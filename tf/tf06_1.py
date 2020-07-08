import tensorflow as tf
tf.set_random_seed(777) # TensorFlow는 서로 다른 분포를 가진 난수 텐서들을 생성하는 여러가지 연산들을 제공합니다. 난수 연산들은 상태를 가지며 , 계산될 때마다 새로운 난수를 생성합니다.

# x_train = [1, 2, 3]
# y_train = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

# random_normal 정규분포로부터의 난수값을 반환합니다.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


sess = tf.Session()
# sess.run(tf.global_variables_initializer()) # 변수를 선언하겠다. 코드하나에 한번만 쓰면 됨
# print(sess.run(W))

hypothesis = x_train*W+b  # y= wx+b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss = mse

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)  # optimizer = 그라디언트 
'''
with tf.Session() as sess:      # with문 쓰면 close 안쓰고 자동으로 닫힘 session 닫기 위한 용도
    sess.run(tf.global_variables_initializer()) # 변수 선언

    for steps in range(2001):                      # 2000 epoch
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={x_train : [1, 2, 3], y_train : [3, 5, 7]}) # train은 하되  _, 결과값 출력하지 않겠다.

        
        if steps % 20 == 0:
            print(steps, cost_val, W_val, b_val)    

'''

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for steps in range(2001):                     
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={x_train : [1, 2, 3], y_train : [3, 5, 7]})

        
        if steps % 20 == 0:
            print(steps, cost_val, W_val, b_val)    

# predict 해보자
    print("예측 : ", sess.run(hypothesis, feed_dict={x_train:[5, 6]}))
    print("예측 : ", sess.run(hypothesis, feed_dict={x_train:[6, 7, 8]}))
