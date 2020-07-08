import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)    # 변하지 않은 상수
node2 = tf.constant(4.0)                # 그대로 출력시 노드의 대한 자료형 출력
node3 = tf.add(node1, node2)

sess = tf.Session()

# 1
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# 2
adder_node = a + b 

# 3 sess.run 출력시 / feed_dict 값 입력 시 
print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1, 3], b:[2, 4]}))

add_and_triple = adder_node*3
print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))


