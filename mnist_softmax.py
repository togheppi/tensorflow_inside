# MNIST example with Softmax Regression

# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# input data: 28x28x1 image, output data: 10 labels of digit from 0 to 9
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 100
epoch = 1000
for i in range(epoch):
    batch = mnist.train.next_batch(batch_size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    if i % 100 == 0:
        print("Epoch: %d, Training accuracy: %g" % (i, accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})))



print("Softmax Regressions with stochastic gradient descent, Loss func. = Cross-entropy:")
print("Training condition: batch size: %d, epoch: %d" % (batch_size, epoch))
print("Test accuracy: %g%%" % (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100))
