# experimenting with pandas 
# Part 1: https://data36.com/pandas-tutorial-1-basics-reading-data-files-dataframes-data-selection/


import numpy as np
import pandas as pd

# load csv creating a pandas dataframe
iris = pd.read_csv('iris.csv', delimiter = ',')

# I want to see all rows in the output https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html
#pd.options.display.max_rows = 151

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



# Part 2: https://data36.com/pandas-tutorial-2-aggregation-and-grouping/

# number of values in each column
#print (iris.count())
# or individual column
#print (iris.species.count())

# to get the sum of a column (below adds up all the values in the sepal_length column)
#print (iris.sepal_length.sum())

# sum of all columns (note how species gets "added" by concatenating all word strings steosasetosasetosa....)
# print(iris.sum())

# minimum and maximum of each column
#print (iris.sepal_length.min())
#print (iris.sepal_length.max())

# mean and median of each column
#print (iris.sepal_length.mean())
#print (iris.sepal_length.median())

# Grouping and Aggregation
# mean below but can be used for other calculations
# mean for each measurement by species
# print(iris.groupby('species'). mean())
# specific measurements below
# Returns Dataframe object (2 dimensional table with columns and rows)
#print (iris.groupby('species').mean()[[('sepal_length')]])
# Returns series object (1 dimensional labelled array)
#print (iris.groupby('species').mean().sepal_length)


# Part 3: https://data36.com/pandas-tutorial-3-important-data-formatting-methods-merge-sort-reset_index-fillna/
# sorts data by petal width from smallest to largest
# could this be used to see if all measurements in a certain species are above/below a certain number?
#print(iris.sort_values('petal_width'))

# sort by multiple columns
#print(iris.sort_values(by = ['species', 'petal_width']))

# to change ascending to descending order
#print(iris.sort_values(by = ['petal_width'], ascending = False))


# https://ugoproto.github.io/ugo_py_doc/Pandas_DataFrame_Notes.pdf
# Results: count, mean, standard deviation, min, max, lower percentile (25%), median (50%) and upper percentile (75%)
#print (iris.describe())


# https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html
# datatypes
# print(iris.dtypes)




