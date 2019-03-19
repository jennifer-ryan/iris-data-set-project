# Machine Learning Project https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# Data Summary

# Load libraries
import pandas
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
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
print( )

# We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.
print(dataset.shape)
print( )

# peek at the data
print(dataset.head(20))
print( )

# Now we can take a look at a summary of each attribute.
# This includes the count, mean, the min and max values as well as some percentiles.
print(dataset.describe())
print( )

# class distribution
# the number of instances (rows) that belong to each class
print(dataset.groupby('class').size())
print( )
