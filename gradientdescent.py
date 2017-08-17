import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import 

w = tf.Variable([.3],dtype=tf.float32)
b = tf.Variable([-.3],dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = w * x + b #modelo

sess = tf.Session()
""" init es un identificador para el subgrafico tf 
	que inicializa todas las variables
"""
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model,{x:[1,2,3,4]}))
y = tf.placeholder(tf.float32)

squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss,{x:[1,2,3,4], y:[0,-1,-2,-3]}))
fixW = tf.assign(w,[-1.])
fixb = tf.assign(b,[1.])
sess.run([fixW,fixb])

print(sess.run(loss,{x:[1,2,3,4], y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)

for i in range(1000):
	sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
print(sess.run([w,b]))