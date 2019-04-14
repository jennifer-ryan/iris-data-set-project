# Machine Learning Project https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# Graphs

import pandas as pd

import matplotlib.pyplot as plt


# Load dataset
iris = pd.read_csv('iris.csv', delimiter = ',')

# univariate plots - to better understand each attribute
# box and whisker plots
iris.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
#iris.hist()
#plt.show()

# multivariate plots - to better understand the relationships between attributes.
# scatter plot matrix
#pd.plotting.scatter_matrix(iris)
#plt.show()
# Note the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.

# **SEPAL AND PETAL SCATTERPLOTS SHOW SETOSA SEPARATION**
#https://github.com/venky14/Machine-Learning-with-Iris-Dataset/blob/master/Iris%20Species%20Dataset%20Visualization.ipynb
# Nice scatterplot of sepal length/width. Saved as scatter.sepal
import seaborn as sns
#sns.FacetGrid(iris, hue = 'species', height=5).map(plt.scatter, 'sepal_length', 'sepal_width').add_legend()
# same below but with petals
#sns.FacetGrid(iris, hue = 'species', height=5).map(plt.scatter, 'petal_length', 'petal_width').add_legend()
#plt.show()

# Swarmplots
# https://seaborn.pydata.org/generated/seaborn.swarmplot.html
sns.swarmplot(x = 'species', y = 'sepal_length', data = iris)
sns.swarmplot(x = 'species', y = 'sepal_width', data = iris)
sns.swarmplot(x = 'species', y = 'petal_length', data = iris)
sns.swarmplot(x = 'species', y = 'petal_width', data = iris)
plt.show()

# Pivot Table
# appended from https://www.kaggle.com/lalitharajesh/iris-dataset-exploratory-data-analysis
#pd.pivot_table(iris, index=['species'], values = ['sepal_length'], aggfunc=[np.mean, np.std])



