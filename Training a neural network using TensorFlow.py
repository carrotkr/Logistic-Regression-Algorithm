import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

#%%
# mnist.train (55,000), mnist.test (10,000), mnist.validation (5,000)
data_mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Image of shape 28*28=784.
image_shape = 784
num_classes = 10

X = tf.placeholder(tf.float32, [None, image_shape])
Y = tf.placeholder(tf.float32, [None, num_classes])

# For nn layers.
weights = tf.Variable(tf.random_normal([image_shape, num_classes]))
bias = tf.Variable(tf.random_normal([num_classes]))

# Multiplies matrix a by matrix b.
hypothesis = tf.matmul(X, weights) + bias

#%%
# Minimize error using softmax cross entropy.
# Reference (tf.nn.softmax_cross_entropy_with_logits).
#   www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

# Gradient Descent.
# Reference (tf.compat.v1.train.GradientDescentOptimizer).
#   www.tensorflow.org/api_docs/python/tf/compat/v1/train/GradientDescentOptimizer
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

#%% Training.
# Initialize.
# Reference (tf.compat.v1.global_variables_initializer).
#   www.tensorflow.org/api_docs/python/tf/compat/v1/global_variables_initializer
initialize = tf.global_variables_initializer()

# Reference (tf.compat.v1.Session).
#   www.tensorflow.org/api_docs/python/tf/compat/v1/Session
sess = tf.Session()
sess.run(initialize)

num_epochs = 10
size_batch = 1000
batch_total = int(data_mnist.train.num_examples / size_batch)

for epoch in range(num_epochs):    
    cost_description = 0
    
    traing_report = tqdm(range(batch_total))
    for i in traing_report:
        # Stochstic Gradient Descent.
        batch_X, batch_Y = data_mnist.train.next_batch(size_batch)
        _, accumulate_cost = sess.run([gradient_descent, cost],\
                                      feed_dict={X: batch_X, Y: batch_Y})
        cost_description += accumulate_cost / batch_total
        traing_report.set_description('Cost: %f' % cost_description)
    
    print('\nEpoch: %d' % (epoch+1), '\nCost: %f' % cost_description)

#%% Test model and calculate accuracy.
# Reference (tf.math.equal).
#   www.tensorflow.org/api_docs/python/tf/math/equal
# Reference (tf.math.argmax).
#   www.tensorflow.org/api_docs/python/tf/math/argmax
prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

# Reference (tf.cast).
#   www.tensorflow.org/api_docs/python/tf/cast
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

print('\nAccuracy: ', accuracy.eval(session=sess,\
                                  feed_dict={X: data_mnist.test.images, Y: data_mnist.test.labels}))