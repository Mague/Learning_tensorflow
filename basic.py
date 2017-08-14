import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

node1 = tf.constant(3.0,dtype=tf.float32)
node2 = tf.constant(2.0)
# print(node1,node2)
sess = tf.Session()
print(sess.run([node1,node2]))
node3 = tf.add(node1,node2)
print("node3: ",node3)
print("sess.run(node3):",sess.run(node3))
node4 = tf.add(node3,node3)
print("node4: ",node4)
print("sess.run(node4):",sess.run(node4))

# Marcador de posicion
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # es igual a tf.add(a,b)
print(sess.run(adder_node,{a:11,b:15.9}))
print(sess.run(adder_node,{a:[11,12],b:[15.9,12]}))
add_and_triple = adder_node * 3
print(sess.run(add_and_triple,{a:11,b:15.9}))