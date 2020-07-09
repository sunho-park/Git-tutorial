# 회귀
from sklearn.datasets import load_diabetes
import tensorflow as tf

# load a csv
# CSV_PATH = './data/.csv'
# dataset = tf.contrib.data.make_csv_dataset(CSV_PATH, batch_size=32)

dataset = load_diabetes()

x_data = dataset.data
y_data = dataset.target

y_data = y_data.reshape(442, 1)
print(x_data.shape)     # (442, 10)
print(y_data.shape)     # (442, 1)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([10, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b 
# 이 자체가 activation = defalut  // linear

cost = tf.reduce_mean(tf.square(hypothesis-y)) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.75) # 0.75 2879 / .4 2883 #0.3 2889 / 0.25 2899/
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _= sess.run(
        [cost, hypothesis, train], feed_dict={x: x_data, y: y_data})
    
    if step % 100==0:
        print(step, "cost:", cost_val, "\n 예측값 : ", hy_val)
