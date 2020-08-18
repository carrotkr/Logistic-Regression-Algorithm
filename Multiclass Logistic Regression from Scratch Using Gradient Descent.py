import pandas as pd # Data processing.
import numpy as np # Linear algebra.
import seaborn as sns # Python graphing library.
sns.set(style='white', color_codes=True)
import matplotlib.pyplot as plt

#%%
# Data location.
data_iris = pd.read_csv('/Users/kbkim/Dropbox/Iris.csv')
print(data_iris.head())

data_iris.drop(['Id'], axis=1, inplace=True)
print(data_iris.head())

print(data_iris['Species'].value_counts())

# Draw a plot of two variables with bivariate and univariate graphs.
sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=data_iris, size=6)

# Multi-plot grid for plotting conditional relationships.
sns.FacetGrid(data_iris, hue='Species', size=6)\
    .map(plt.scatter, 'SepalLengthCm', 'SepalWidthCm').add_legend()

sns.jointplot(x='PetalLengthCm', y='PetalWidthCm', data=data_iris, size=6)

sns.FacetGrid(data_iris, hue='Species', size=6)\
    .map(plt.scatter, 'PetalLengthCm', 'PetalWidthCm').add_legend()
    
#%%
# Logistic regression.
def sigmoid_function(x):
    return 1 / (1+np.exp(-x))

# Logistic regression cost function (log loss).
def cost_function(theta, x, y):
    # Estimated probability.
    p = sigmoid_function(x @ theta)
    
    cost = -(1/len(y)) * (np.sum((y*np.log(p)) + ((1-y)*np.log(1-p))))
    
    return cost

def gradient_function(theta, x, y):
    # Estimated probability.
    p = sigmoid_function(x @ theta)
    
    gradient = (1/len(y)) * ((y-p) @ x)
    
    return gradient
    
#%%
# Iris-setosa = 0, Iris-versicolor = 1, Iris-virginica = 2.
data_iris['Species'] = data_iris['Species'].astype('category').cat.codes

data = []
data = np.array(data_iris)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = \
    train_test_split(data[:, [0,1,2,3]], data[:, 4], test_size=0.4, random_state=33)

plt.scatter(X_train[:, 2], X_train[:, 3], c=Y_train)
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.show()

#%% Training.
learning_rate = 0.05
num_epochs = 5000

num_classes = np.unique(Y_train)
collect_theta = []
cost = np.zeros(num_epochs)

X_train = np.insert(X_train[:, 2:], 0, 1, axis=1)

for i in num_classes:
    # Reference.
    #   scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
    # One-vs-the-rest (OvR) multiclass/multilabel strategy
    OvR = np.where(Y_train == i, 1, 0)
    
    theta = np.zeros(X_train.shape[1])
    
    for epoch in range(num_epochs):
        cost[epoch] = cost_function(theta, X_train, OvR)
        gradient = gradient_function(theta, X_train, OvR)
        theta += learning_rate * gradient
    
        if (epoch%1000 == 0):
            print('class: %1d' % (i))            
            print('epoch:', epoch)

            plt.scatter(X_train[:, 1], X_train[:, 2], c=Y_train)
            plt.title('Classify for Class %1d' % (i))
            plt.xlabel('PetalLengthCm')
            plt.ylabel('PetalWidthCm')
            
            x = np.linspace(1, 7)
            y = -(theta[0] + (theta[1]*x)) / theta[2]
            plt.plot(x, y, color='red')
            plt.show()
    
    collect_theta.append(theta)
    
plt.scatter(X_train[:, 1], X_train[:, 2], c=Y_train)
plt.title('Decision Boundary')
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')

for theta in (collect_theta[0], collect_theta[2]):
    x = np.linspace(1, 7)
    y = -(theta[0] + (theta[1]*x)) / theta[2]
    plt.plot(x, y, color='red')
    
#%%
plt.plot(cost)
plt.title('Cost Function Value vs Iteration of Gradient Descent')
plt.xlabel('Number of Epochs')
plt.ylabel('Costs')
plt.show()
