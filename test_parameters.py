# Created to test different values for K and check test_size figure.

# Import required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn import metrics
import numpy as np

# Load dataset
iris = datasets.load_iris()

# Feature and label differentiation 
x = iris.data
y = iris.target

# Set initial K value
k = 3

# Checks from 3 - 23 to assess best fit K value for the model. 
# Iterates through KNN model creation to find best fit.
while k < 23:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) # 0.2 value changed to 0.3 to test 30% test size also
    knn = knc(n_neighbors = k)
    knn.fit(x_train, y_train)
    test_prediction = knn.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, test_prediction)
    print(k, "=", accuracy)
    k += 2
# Code run through 10 times with test_size set to 0.2 and a further 10 times with test_size set to 0.3
# Results saved to tables. See Images/k_values.PNG 