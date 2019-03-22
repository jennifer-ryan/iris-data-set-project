# Help from https://stackoverflow.com/a/29432741

import matplotlib.pyplot as plt
import pandas

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)



plt.matshow(dataset.corr())
plt.show()