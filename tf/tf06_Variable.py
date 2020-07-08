import tensorflow as tf
tf.set_random_seed(777) 

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
print(W)

W = tf.Variable([0.3], tf.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer()) 
aaa = sess.run(W)
print(aaa)
sess.close() # 반드시 해야함


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer()) # same
bbb = W.eval()
print(bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print(ccc)
sess.close()


