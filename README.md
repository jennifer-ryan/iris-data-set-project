# iris-data-set-project


## The Dataset
Slight discrepancies have been noted in some electronic versions of the dataset (Bezdek et al, 1999) with slight variations observed in some of the measurements when compared to the original dataset that was published in 1936.
 
iris.csv taken from [here](https://gist.github.com/curran/a08a1080b88344b0c8a7#file-iris-csv-L1).

Notes
Wiki:
Quantifies the morphology (structural features) of three related species of iris flowers
Multivariate data set - more than 2 variables per observation
Linear Discriminant Analysis / Fisher's Linear Discriminant - a linear combination that characterises/separates two or more classes of objects.

Try working through this tutorial to explore dataset: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
A machine learning project may not be linear, but it has a number of well known steps:
1. Define Problem.
2. Prepare Data.
3. Evaluate Algorithms.
4. Improve Results.
5. Present Results.

The best small project to start with on a new tool is the classification of iris flowers (e.g. the iris dataset). This is a good project because it is so well understood.
* Attributes are numeric so you have to figure out how to load and handle data.
* It is a classification problem, allowing you to practice with perhaps an easier type of supervised learning algorithm.
* It is a multi-class classification problem (multi-nominal) that may require some specialized handling.
* It only has 4 attributes and 150 rows, meaning it is small and easily fits into memory (and a screen or A4 page).
* All of the numeric attributes are in the same units and the same scale, not requiring any special scaling or transforms to get started.




Look at: 
* Classification - organising data by categories
* Linear discriminant analysis 
* Data visualisation - graphs
* Pattern recognition - species could be predicted based on data measurements
* Machine learning such as neural networks - predictive nature of the data 


## Averages
Simply observing the averages of each measurement by species seems to demonstrate significant differences between the species. For instance, setosa has the smallest sepal length, petal length and petal width measurements and yet the largest sepal widths.
None of the species individual averages are close to the overall average for all species.

## Standard Deviation
How close measurements are the to average - how wide is the bell curve?


## Scatterplots 
![Petal Measurements](scatter.petal.png)

## Correlations
Correlation figures demonstrate that petal length and petal width are both highly correlated with sepal length
Petal length and width are very highly correlated
However, sepal length and width have a small negative correlation.



## References
Bezdek, J. C., Keller, J. M., Krishnapuram, R., Kuncheva, L. I., & Pal, N. R.        (1999). *Correspondence: Will the Real Iris Data Please Stand Up?*. IEEE        Transactions on Fuzzy Systems, 7: 3, June 1999.
