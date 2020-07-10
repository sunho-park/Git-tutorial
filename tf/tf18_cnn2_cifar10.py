from keras.datasets import cifar10 
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)        # (50000, 32, 32, 3)
print(y_train.shape)        # (50000, 1)
print(x_test.shape)         # (10000, 32, 32, 3)
print(y_test.shape)         # (10000, 1)

#  one - hot
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train.astype('int32'))
y_test = np_utils.to_categorical(y_test.astype('int32'))

x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/ batch_size)     
keep_prob = tf.placeholder(tf.float32)

# 
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])

x_img = tf.reshape(x, [-1, 32, 32, 3])

# 레이어1
w1 = tf.get_variable("w1", shape=[5, 5, 3, 64]) # Conv2D(64, (5, 5), input_shape = (32, 32, 3)) 
L1 = tf.nn.conv2d(x_img, w1, strides=[1, 1, 1, 1], padding='SAME')
print('w1 :', w1)       # shape=(5, 5, 3, 64)
print('L1 : ', L1)      # shape=(?, 32, 32, 64)
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print('L1 : ', L1)      # shape=(?, 16, 16, 64)
# 레이어2
w2 = tf.get_variable("w2", shape=[3, 3, 64, 32])
L2 = tf.nn.conv2d(L1, w2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print('L2 ', L2)        # shape=(?, 8, 8, 32)

# 레이어3
L2_flat = tf.reshape(L2, [-1, 8*8*32])
w3 = tf.get_variable("w3", shape=[8*8*32, 10], initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))
print('L2_flat : ', L2_flat)    # shape=(?, 2048)

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




