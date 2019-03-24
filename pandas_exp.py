# experimenting with pandas https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html
# https://data36.com/pandas-tutorial-1-basics-reading-data-files-dataframes-data-selection/

import numpy as np
import pandas as pd

# load csv creating a pandas dataframe
iris = pd.read_csv('iris.csv', delimiter = ',')

# prints first 5 entries
#print(iris.head())

# prints last 5 entries
#print(iris.tail())

# prints 10 random entries
#print(iris.sample(10))

# prints select particular columns
#print (iris[['sepal_length', 'species']])

# prints rows filtered by type. this could be useful when trying to separate species for analysis 
#print(iris[iris.species == 'setosa'])

# prints means of a numerical columns 
#print(iris.mean())

# prints mean of a particular column. https://stackoverflow.com/a/31037360
#print(iris["sepal_width"].mean())

# splitting by species
# setosa = iris['species'] == 'setosa'
#print(setosa) # prints as boolean true/false

