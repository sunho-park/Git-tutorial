import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1, 2], 
          [2, 3], 
          [3, 1], 
          [4, 3], 
          [5, 3], 
          [6, 2]]

y_data = [[0], 
          [0], 
          [0], 
          [1],
          [1], 
          [1]]

x = tf.placeholder(tf.float32, shape=[None, 2]) # summary에서 봄 input_shape=(3, )
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # activation = 'sigmoid'

# cost = tf.reduce_mean(tf.square(hypothesis-y)) 

cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis)) # binary crossentrophy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # 1e-5 = 0.00001
train = optimizer.minimize(cost)

# tf.cast       
'''
입력한 값의 결과를 지정한 자료형으로 변환해줌
x = tf.constant([1.8, 2.2], dtype=tf.float32)
tf.cast(x, tf.int32)
# 출력값 : [1, 2], dtype=tf.int32
입력한 값의 결과를 변환해주므로 조건절을 넣을 수 있음(True/False 반환)
tf.cast(x > 2, dtype=tf.float32)
# 출력값 : [ 0.,  1.], dtype=float32'''

# predict
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y),    # tf.equal(x, y) : x, y를 비교하여 boolean 값을 반환 값이 같으면 true 틀리면 false / tf.not_equal   
                            dtype=tf.float32))

# 여기 까지 붕어빵 틀을 만들어 놓은것 뿐 / 실행은 sess.run


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        if step % 200 == 0:
            print(step, cost_val)
    
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={x:x_data, y:y_data})

    print("\n Hypothesis : ", h, "\n Correct (y) : ", 
            "\n Accuracy : ", a)







