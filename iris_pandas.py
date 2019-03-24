# experimenting with pandas https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html

import pandas as pd
import numpy as np

# Load dataset. Better to use pandas rather than numpy for this as the whole array in numpy need to be of the same datatype
data = pd.read_csv('iris.csv', delimiter=',')

# shows averages for each different measurement
print(np.mean(data))

