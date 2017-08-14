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