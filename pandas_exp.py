# experimenting with pandas https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html
# https://data36.com/pandas-tutorial-1-basics-reading-data-files-dataframes-data-selection/


import numpy as np
import pandas as pd

# load csv creating a pandas dataframe
iris = pd.read_csv('iris.csv', delimiter = ',')

# first 5 entries
#print(iris.head())

# last 5 entries
#print(iris.tail())

# 10 random entries
#print(iris.sample(10))

# select particular columns
#print (iris[['sepal_length', 'species']])

# rows filtered by type. this could be useful when trying to separate species for analysis 
#print(iris[iris.species == 'setosa'])

# means of a numerical columns 
#print(iris.mean())

# mean of a particular column. https://stackoverflow.com/a/31037360
#print(iris["sepal_width"].mean())

# splitting by species
# setosa = iris['species'] == 'setosa'
#print(setosa) # prints as boolean true/false



# https://data36.com/pandas-tutorial-2-aggregation-and-grouping/

# number of values in each column
#print (iris.count())
# or individual column
#print (iris.species.count())

# to get the sum of a column (below adds up all the values in the sepal_length column)
#print (iris.sepal_length.sum())

# sum of all columns (note how species gets "added" by concatenating all word strings steosasetosasetosa....)
# print(iris.sum())

# minimum and maximum of each column
print (iris.sepal_length.min())
print (iris.sepal_length.max())

# mean and median of each column
print (iris.sepal_length.mean())
print (iris.sepal_length.median())

# Grouping