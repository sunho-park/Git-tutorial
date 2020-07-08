# 이진 분류
from sklearn.datasets import load_diabetes
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, ])
y = tf.placeholder(tf.float32, shape=[None, ])




