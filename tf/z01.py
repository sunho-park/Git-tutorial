import tensorflow as tf
print(tf.__version__)

hello = tf.constant("Hello Sidney")

sess = tf.Session()

print(sess.run(hello))



