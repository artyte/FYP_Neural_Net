#example of a 0 hidden layer neural net
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
epoch = 10000
learning_rate = 0.001
display_step = 100

x = tf.placeholder(tf.float32, [None, n_input])	#input
W = tf.Variable(tf.zeros([n_input, n_classes]))	#weight
b = tf.Variable(tf.zeros([n_classes]))	#bias
y = tf.nn.softmax(tf.matmul(x, W) + b)	#predicted output
y_ = tf.placeholder(tf.float32, [None, 10])	#actual output

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_)) 	#cross entropy cost
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)	#stochastic gradient descent

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(epoch):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	if (i+1)% display_step == 0:
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(y_,1)), tf.float32))
		print("Accuracy = ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))