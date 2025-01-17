# 이진 분류
from sklearn.datasets import load_breast_cancer
import tensorflow as tf

datasets = load_breast_cancer()

x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(569, 1)

print(x_data.shape)     # (569, 30)
print(y_data.shape)     # (569, )

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.zeros([30, 1], name='weight'))
b = tf.Variable(tf.zeros([1], name='bias'))

# zeros 와 random_normal 의 차이?
print("w : ", w)

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))

# binary_crossentrophy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000002) 
train = optimizer.minimize(cost)

# predict
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y),     
                            dtype=tf.float32))


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

# zeros 와 random_normal 의 차이?




