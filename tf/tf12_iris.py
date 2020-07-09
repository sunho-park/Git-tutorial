# 다중 분류
# iris 코드를 완성하시오.
from sklearn.datasets import load_iris
import tensorflow as tf

datasets = load_iris()

x_data = datasets.data
y_data = datasets.target

# print(x_data)
print(x_data.shape)  #  (150, 4)
print(y_data.shape)  #  (150, )

# 원핫 인코딩
y_onehot = tf.one_hot(y_data, 3)
y_onehot = tf.reshape(y_onehot, [-1, 3])

sess = tf.Session()
y_hot = sess.run(y_onehot)

print("one_hot", sess.run(y_onehot)) 
print(y_hot)

x = tf.placeholder('float32', shape=[None, 4])
y = tf.placeholder('float32', shape=[None, 3])

w = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')


hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_onehot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss],
                                    feed_dict = {x:x_data, y:y_hot})

        if step % 200 == 0:
            print(step, cost_val)
    h, c, a = sess.run([hypothesis, prediction, accuracy],
                                feed_dict={x:x_data, y:y_hot})
    print("\n Hypothesis : ", h, "\n Correct (y) : ", 
            "\n Accuracy : ", a)
