import tensorflow as tf
# 3+4+5 
# 4-3
# 3*4
# 4/2

a = tf.constant(3)
b = tf.constant(4)
c = tf.constant(5)

sum = a + b + c
min = b - a
mul = a*b
dvi = b/2

sess = tf.Session()

print("1. 덧셈 ) A+B =", sess.run(sum))
print("2. 뺼셈 ) A-B =", sess.run(min))

print("3. 곱하기 ) A*B =", sess.run(mul))
print("4. 나누기 ) A/B =", sess.run(dvi))
