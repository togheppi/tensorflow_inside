# MNIST example with Convolutional Neural Network (CNN)

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


# initiate random weight variable of normal distribution
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# initiate random bias variable of constant ( = 0.1)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 2D convolution with [1,1] stride, output size is same as input
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 2x2 max pooling function with [2,2] stride, output size is same as input
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# First convolution layer (5x5, 32 features)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])  # reshape x as 28x28x1 image for 2d convolution
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# First pooling layer (max pooling 2x2)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolution layer (5x5, 64 features)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# Second pooling layer (max pooling 2x2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully-connected layer (1024 neurons)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout for prohibiting overfitting

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # final model (y = W*x + b)

# Loss function & optimizer
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  # loss function: cross_entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # optimizer: Adam
# Evaluation
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

batch_size = 50
epoch = 200
for i in range(epoch):
    batch = mnist.train.next_batch(batch_size)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("Convolution Neural Networks of 2 layers with Adam optimizer, Loss func.= Cross-entropy:")
print("Training condition: batch size: %d, epoch: %d" % (batch_size, epoch))
print("Test accuracy: %g%%" %(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100))

