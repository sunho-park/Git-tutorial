# hypothesis를 구하시오
# H = Wx + b
# aaa, bbb, ccc 자리에 각 hypothesis를 구하시오.
import tensorflow as tf
tf.set_random_seed(777) 

x = [1, 2, 3]
# W = tf.Variable([0.3], tf.float32)
W = tf.Variable([0.3])
b = tf.Variable([1.])


hypothesis = W*x + b

sess = tf.Session()
sess.run(tf.global_variables_initializer()) 
aaa = sess.run(hypothesis)
print('aaa : ', aaa)  # < - 요기 완성
sess.close() # 반드시 해야함

'''
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer()) # same
bbb = W.eval()
print('hypothesis : ', hypothesis)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print('hypothesis : ', hypothesis)
sess.close()
'''

