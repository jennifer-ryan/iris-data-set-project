# Machine Learning Project https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# Data Summary

#Go back to this:
# guide to exploring datasets https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/
# selecting subsets https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c
# exploration https://medium.com/@harimittapalli/exploratory-data-analysis-iris-dataset-9920ea439a3e

# Steps in data exploration 
# https://datahero.com/blog/2013/11/20/5-beginners-steps-to-investigating-your-dataset/
# https://www.quora.com/What-are-the-steps-include-in-data-exploration


# Load libraries
import pandas as pd


# Load dataset
iris = pd.read_csv('iris.csv', delimiter = ',')

# We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.
print(iris.shape)


# peek at the data
print(iris.head(20))

# Now we can take a look at a summary of each attribute.
# This includes the count, mean, the min and max values as well as some percentiles.
print(iris.describe())


# class distribution
# the number of instances (rows) that belong to each class
print(iris.groupby('species').size())


# data types
iris.dtypes

# splitting the dataset by species
# help from https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c

index = iris.index
columns = iris.columns
values = iris.values

setosa = iris.loc[0:49]
versicolor = iris.loc[50:99]
virginica = iris.loc[100:149]

