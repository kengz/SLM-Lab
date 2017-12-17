'''
Newbie tensorflow refresher
'''
from slm_lab.lib import util, viz
import numpy as np
import pandas as pd
import pydash as _
import tensorflow as tf
# import tensorflow.contrib.eager as tfe

# tfe.enable_eager_execution()

# constant
n1 = tf.constant(3.0, dtype=tf.float32)
n2 = tf.constant(4.0)

sess = tf.Session()
sess.run([n1, n2])

n3 = n1 + n2
sess.run(n3)

# placeholder: like stdin to feed data
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = a + b
sess.run(c, {a: 3, b: 4.5})
sess.run(c, {a: [1, 3], b: [2, 4]})
d = c * 3
sess.run(d, {a: 3, b: 4.5})

# Variable: variable with type and init value
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
sess.run(init)

# run and eval loss
sess.run(linear_model, {x: [1, 2, 3, 4]})
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

# fix to minimize loss
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

# use optimizer to min-ize loss
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

sess.run([W, b, loss], {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})


# estimator: frontend to run both training and eval
feature_columns = [tf.feature_column.numeric_column('x', shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_fn, steps=1000)
train_metrics = estimator.evaluate(input_fn=input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
train_metrics
eval_metrics


# MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/mnist/', one_hot=True)

# None means it can be any len later
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# y = tf.softmax(tf.matmul(x, W) + b)
ylogits = tf.matmul(x, W) + b
y = tf.nn.softmax(ylogits)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=ylogits)
# cross_entropy = tf.reduce_mean(cross_entropy) * 100

# # custom gradient
# grad_W, grad_b = tf.gradients(xs=[W, b], ys=cross_entropy)
# new_W = W.assign(W - lr * grad_W)
# new_b = b.assign(b - lr * grad_b)
# then run sess.run([new_W, new_b, cross_entropy]) instead of sess.run(train)

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

is_correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


# Conv MNIST
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/mnist/', one_hot=True, reshape=False)

# to build graph on the fly
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# the model
# channels
c0 = 1
c1 = 6
c2 = 12
fc_size = 200  # fully connected before output
stride = 1

W1 = weight_variable([6, 6, c0, c1])
b1 = bias_variable([c1])

h_conv1 = tf.nn.relu(conv2d(x, W1) + b1)
h_pool1 = max_pool_2x2(h_conv1)

W2 = weight_variable([5, 5, c1, c2])
b2 = bias_variable([c2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
h_pool2 = max_pool_2x2(h_conv2)

# fully connected
W3 = weight_variable([7 * 7 * c2, fc_size])
b3 = bias_variable([fc_size])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * c2])
h_3 = tf.nn.relu(tf.matmul(h_pool2_flat, W3) + b3)

# dropout
h_drop = tf.nn.dropout(h_3, pkeep)

W_out = weight_variable([fc_size, 10])
b_out = bias_variable([10])

y_conv = tf.matmul(h_drop, W_out) + b_out

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
optimizer = tf.train.AdamOptimizer(lr)
train = optimizer.minimize(cross_entropy)
is_correct = tf.equal(tf.argmax(y_conv, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={x: batch[0], y_: batch[1], pkeep: 1.0})
            print(f'train acc: {i}, {train_accuracy}')
        train.run(feed_dict={x: batch[0], y_: batch[1], pkeep: 0.5, lr: 0.01})

    train_accuracy = accuracy.eval(
        feed_dict={x: batch[0], y_: batch[1], pkeep: 1.0})
    print(f'train acc: {i}, {train_accuracy}')
