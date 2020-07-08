from sklearn.datasets import load_diabetes
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 8])
y = tf.placeholder(tf.float32, shape=[None, 1])

dataset = tf.load