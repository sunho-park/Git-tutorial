import tensorflow as tf
tf.set_random_seed(777) 
# TensorFlow는 서로 다른 분포를 가진 난수 텐서들을 생성하는 여러가지 연산들을 제공합니다. 난수 연산들은 상태를 가지며 , 계산될 때마다 새로운 난수를 생성합니다.

x_train = [1, 2, 3]
y_train = [3, 5, 7]

# random_normal 정규분포로부터의 난수값을 반환합니다.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


sess = tf.Session()
# sess.run(tf.global_variables_initializer()) # 변수는 항상 초기화
# print(sess.run(W))

hypothesis = x_train*W+b  # y= wx+b
sess.run(tf.global_variables_initializer()) 
sess.run(hypothesis)
print(hypothesis)

'''
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss = mse / loss = abs tf.abs()

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)  # sgb / optimizer  tf.train.AdagradOptimizer

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수 선언

    for steps in range(2001):                   # 2001 epoch
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b]) # train은 하되  _, 결과값 출력하지 않겠다.
        
        if steps % 20 == 0:
            print(steps, cost_val, W_val, b_val)    

'''