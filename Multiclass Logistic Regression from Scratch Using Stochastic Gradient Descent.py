import numpy as np # Linear algebra.
import pandas as pd # Data processing.
import matplotlib as mpl
import matplotlib.pyplot as plt

#%%
data_mnist = pd.read_csv('/Users/carrotkr/Dropbox/MNIST_train.csv')

X_train = data_mnist.drop(labels=["label"], axis=1).to_numpy()
Y_train_label = data_mnist['label']
print(X_train.shape)
print(Y_train_label.shape)

example_digit = X_train[33]
# Display data as an image on a 2D regular raster.
plt.imshow(example_digit.reshape(28, 28), cmap=mpl.cm.binary)
print(Y_train_label[33])

print(Y_train_label.unique())
print(Y_train_label.value_counts())

num_classes = len(Y_train_label.unique())
Y_train = np.zeros((X_train.shape[0], num_classes))
data_mnist_matrix = data_mnist.to_numpy()
for i in range(num_classes):
    Y_train[:, i] = np.where(data_mnist_matrix[:, 0] == i, 1, 0)

#%%
# Logistic regression.
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def cost_function(theta, x, y):
    # Estimated probability.
    p = sigmoid_function(x @ theta)
    
    cost = -(1/len(y)) * (np.sum(y*np.log(p) + (1-y)*np.log(1-p)))
    
    return cost

def gradient_function(theta, x, y):
    # Estimated probability.
    p = sigmoid_function(x @ theta)
    
    gradient = (1/len(y)) * ((y-p) @ x)
    
    return gradient