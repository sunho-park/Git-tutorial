import tensorflow as tf
import numpy as np

dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(dataset.shape)        # (10, )

# RNN 모델을 짜시오

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq)- size):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    return np.array(aaa)
size = 5

x_data = split_x(dataset, size)
print(x_data) 
print(x_data.shape)  #(5,5)
y_data = dataset[5:]
print(y_data) 
print(y_data.shape)  # (5, )
x_data = x_data.reshape(1, 5, 5)
y_data = y_data.reshape(1, 5)

sequence_length = 5
input_dim = 5
output = 5
batch_size = 1 # 전체 행
output = 5

x = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
y = tf.compat.v1.placeholder(tf.float32, (None, sequence_length))  

# 2. 모델구성
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
print('hypothesis : ', hypothesis)  # Tensor("rnn/transpose_1:0", shape=(?, 5, 5), dtype=float32)
print('_states : ', _states)        # shape=(?, 5) dtype=float32

# 3. 컴파일
cost = tf.reduce_mean(tf.square(hypothesis-y)) 

train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# 3. 훈련
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(101):
    cost_val, hy_val, _= sess.run(
        [cost, hypothesis, train], feed_dict={x: x_data, y: y_data})
    
    if step % 100==0:
        print(step, "cost:", cost_val, "\n 예측값 : ", hy_val)
