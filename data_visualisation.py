# Machine Learning Project https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# Graphs

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
iris = pd.read_csv('iris.csv', delimiter = ',')

# univariate plots - to better understand each attribute
# box and whisker plots
#iris.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

# histograms
#iris.hist()
#plt.show()

# multivariate plots - to better understand the relationships between attributes.
# scatter plot matrix
#scatter_matrix(iris)
#plt.show()
# Note the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.

# **SEPAL AND PETAL SCATTERPLOTS SHOW SETOSA SEPARATION**
#https://github.com/venky14/Machine-Learning-with-Iris-Dataset/blob/master/Iris%20Species%20Dataset%20Visualization.ipynb
# Nice scatterplot of sepal length/width. Saved as scatter.sepal
import seaborn as sns
sns.FacetGrid(iris, hue = 'species', height=5).map(plt.scatter, 'sepal_length', 'sepal_width').add_legend()
# same below but with petals
sns.FacetGrid(iris, hue = 'species', height=5).map(plt.scatter, 'petal_length', 'petal_width').add_legend()
plt.show()
