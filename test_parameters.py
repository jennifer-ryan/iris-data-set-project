# Created to test different values for K
# Also use to check test_size figure? 20%/30% - still undecided

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn import metrics
import numpy as np

iris = datasets.load_iris()

x = iris.data
y = iris.target

k = 3

while k < 23:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    knn = knc(n_neighbors = k)
    knn.fit(x_train, y_train)
    test_prediction = knn.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, test_prediction)
    print(k, '=', accuracy)
    k += 2
