# Demonstrating distinct measurements between species

# import pandas and numpy
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Load iris dataset
iris = pd.read_csv('iris.csv', delimiter = ',')

# average values by species
sp_mean = iris.groupby('species').mean()

# total average values across all species
to_mean = iris.mean()

# creating one single table containing species and total averages
# appending total values to species values https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html
means = sp_mean.append(to_mean, ignore_index=True)

# several attempts to add names of species to the index column
# means.index.names = [...] ValueError
# means.rename(index ={...}) Names not showing up
# Inserting new column rather than trying to rename the index:
# could not get to work until I added square brackets around the species names
means.insert(0, 'species', ['setosa', 'versicolor', 'virginica', 'all_species'])
#print(means)

# Tried prettytable and plotly to display a table - not available. Will stick to matplotlib and seaborn for now


# Same thing for medians. Any use?
#sp_median = iris.groupby('species').median().
#to_median = iris.median()
#medians = sp_median.append(to_median, ignore_index=True)
#medians.insert(0, 'species', ['setosa', 'versicolor', 'virginica', 'all_species'])
#print(medians)

# standard deviation
stan = iris.groupby('species').std()
#print(stan)


# Correlations
cor = iris.corr()
print(cor)