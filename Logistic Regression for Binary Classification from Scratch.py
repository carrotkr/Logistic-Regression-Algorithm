import numpy as np # Linear Algebra
import pandas as pd # Data Processing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Data Location.
data_train_transaction = pd.read_csv('/Users/carrotkr/Downloads/train_transaction.csv')
data_train_identity = pd.read_csv('/Users/carrotkr/Downloads/train_identity.csv')

# Data Location.
data_test_transaction = pd.read_csv('/Users/carrotkr/Downloads/test_transaction.csv')
data_test_identity = pd.read_csv('/Users/carrotkr/Downloads/test_identity.csv')

# Data Location.
sample_submission = pd.read_csv('/Users/carrotkr/Downloads/sample_submission.csv')

print(data_train_transaction.shape)
print(data_train_transaction.info())
print(data_train_transaction.head())
print(data_train_transaction.isna().sum())

print(data_train_transaction.iloc[:, 1].value_counts())
print(data_train_transaction.iloc[:, 1].value_counts() / data_train_transaction.iloc[:,1].count() * 100)

#%% Distribution of Transactions in Train Data
print('{:.4f}% of Transactions are fraud in train data.'\
      .format(data_train_transaction['isFraud'].mean() * 100))

data_train_transaction.groupby('isFraud').count()['TransactionID']\
    .plot(kind='barh', title='Distribution of Transactions in Train Data', figsize=(10, 4))
plt.show()

#%% X_train, X_test, y_train
X_train = data_train_transaction.drop('isFraud', axis=1)
X_test = data_test_transaction.copy()

# Label Encoding
# Reference:
#   https://www.kaggle.com/jesucristo/fraud-complete-eda/notebook
for i in X_train.columns:
    if X_train[i].dtype=='object' or X_test[i].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[i].values) + list(X_test[i].values))
        X_train[i] = lbl.transform(list(X_train[i].values))
        X_test[i] = lbl.transform(list(X_test[i].values))

y_train = data_train_transaction['isFraud'].astype("uint8").copy()

del data_train_transaction, data_test_transaction        

print(X_train.head())
print(X_test.head())
print(y_train.head())

#%% Drop TransactionDT
drop_col = ['TransactionDT']
X_train.drop(drop_col, axis=1, inplace=True)
X_test.drop(drop_col, axis=1, inplace=True)

print(X_train.head())
print(X_test.head())
  
#%% Fill NaN
X_train.fillna(-1, inplace=True)
X_test.fillna(-1, inplace=True)

print(X_train.head())
print(X_test.head())

#%% PCA
# Reference:
#   https://www.kaggle.com/jesucristo/fraud-complete-eda/notebook
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train_PCA = PCA(2).fit_transform(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)         
X_test_PCA = PCA(2).fit_transform(X_test_scaled)

# X_train
plt.title('X_train')
plt.scatter(X_train_PCA[:, 0], X_train_PCA[:, 1], c=y_train)
plt.colorbar()
plt.show()

# X_train
plt.title('X_train')
sns.set_style('white')
sns.scatterplot(X_train_PCA[:, 0], X_train_PCA[:, 1])
plt.show()

# X_test
y_test = np.zeros(X_test.shape[0])
plt.title('X_test')
plt.scatter(X_test_PCA[:, 0], X_test_PCA[:, 1], c=y_test)
plt.colorbar()
plt.show()

#%%
# Logistic Regression for Binary Classification
def sigmoid_function(x):
    return 1 / (1+np.exp(-x))

# Cross Entropy Cost Function (Log Loss)
def cost_function(hypothesis, y):
    cost = (((-y) * np.log(hypothesis)) - ((1-y) * np.log(1-hypothesis))).mean()
    return cost

def gradient_descent(X, hypothesis, y):
    return (np.dot(X.T, (hypothesis-y))) / (y.shape[0])

#%%
X_train = np.concatenate((np.ones((len(X_train_PCA), 1)), X_train_PCA), axis=1)
theta = np.zeros(X_train.shape[1])
cost = []

num_iteration = 1500
learning_rate = 0.03

for i in range(num_iteration):    
    hypothesis = sigmoid_function(X_train @ theta)
    
    cost.append(cost_function(hypothesis, y_train))
    
    # Gradient Descent.
    gradient = gradient_descent(X_train, hypothesis, y_train)
    
    # Gradient Descent Step.
    theta = theta - (learning_rate*gradient)
    
    if ((i%100) == 0):
       sns.scatterplot(X_train[:, 1], X_train[:, 2])
       x = np.linspace(80, 130, 50)
       y = -(theta[0] + (theta[1]*x)) / theta[2]
       
       # Decision Boundary.
       print('Iteration: ', i)
       plt.plot(x, y, color='red')
       plt.title('Decision Boundary')
       plt.show()

# Cost Function Value L vs Iteration of Gradient Descent.
plt.plot(cost)
plt.title('Cost Function Value L vs Iteration of Gradient Descent')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.show()

#%%
def predict(X, parameters):
    return np.round(sigmoid_function(X @ parameters))

parameters_optimal = gradient

X_test = np.concatenate((np.ones((len(X_test_PCA), 1)), X_test_PCA), axis=1)

y_predict = np.zeros(len(X_test))
y_predict = predict(X_test, parameters_optimal)

hypothesis_predict = 1 - sigmoid_function(X_test @ parameters_optimal)

#%%
# Data Location.
subission = pd.read_csv('/Users/carrotkr/Downloads/sample_submission.csv', index_col='TransactionID')
subission['isFraud'] = hypothesis_predict
subission.head()
subission.to_csv('submission.csv')

#%%
# Reference:
#   https://www.kaggle.com/jesucristo/fraud-complete-eda/notebook
subission.head().loc[subission.head()['isFraud'] > 0.99 , 'isFraud'] = 1
b = plt.hist(subission['isFraud'], bins=50)