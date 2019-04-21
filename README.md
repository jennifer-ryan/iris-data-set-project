# Iris Dataset Project 2019
This repository contains an explanation and exploration of the famous Iris Dataset as part of the assessment in the  Programming and Scripting module for the Higher Diploma on Data Analytics with Galway-Mayo Institute of Technology.
 
## About this Repository
The repository is made up of the following:
* This **README** file that contains a description of the Iris Dataset, exploratory data analysis using statistics and visualisation, and a basic illustration of how the dataset can be used in Machine Learning.
* The **iris.csv** file which contains the completed dataset downloaded from [here](https://gist.github.com/curran/a08a1080b88344b0c8a7#file-iris-csv-L1).
* Two **Jupyter Notebooks** created using *Python*: 
    * **Iris Dataset Exploratory Data Analysis** uses the *pandas* library for statistical investigations and *matplotlib* and *seaborn* for data visualisation.
    * **Machine Learning** that interprets the dataset as *numpy* arrays and builds a basic model using *sklearn*, specifically the *K-Nearest Neighbor* algorithm. 
* **Images** folder that contains .png files of some of the data visualisation performed in the Exploratory Data Analysis notebook that are embedded in the README 

## 1. The Dataset
The Iris Dataset consists of 50 samples each of three different species of iris flower: *setosa*, *versicolor* and *virginica*. It contains four different centimetre measurements for each sample - sepal length and width and petal length and width - making it a multivariate dataset.

![Iris Species](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png)

The data was collected by botanist Edgar Anderson in the Gaspé Peninsula and popularised when it was used by biologist and statistician Ronald Fisher in his 1936 paper *The Use of Multiple Measurements in Taxonomic Problems* to demonstrate how statistics can be used for classification. He argues that, based on the information in this dataset, iris group membership could be determined by sepal and petal attributes alone - a method that would become known as linear discriminant analysis. 

It is a 

Slight discrepancies have been noted in some electronic versions of the dataset (Bezdek et al, 1999) with slight variations observed in some of the measurements when compared to the original dataset that was published in 1936.
 

See what is available in this article: https://link.springer.com/chapter/10.1007/978-3-7908-1883-3_19

Why the iris dataset?
* It is a complete, balanced dataset in that there are no null values and each class is equally represented. 
* Each of the four features (sepal and petal length and width) are measured in the same units (centimetres).

It is often used as an example of machine learning because prediction is easy

## 2. Exploratory Data Analysis
https://medium.com/@harimittapalli/exploratory-data-analysis-iris-dataset-9920ea439a3e
Exploratory Data Analysis allows us to better understand the data through statistical and visual techniques in order to form hypotheses

## Averages
Simply observing the averages of each measurement by species seems to demonstrate significant differences between the species. For instance, setosa has the smallest sepal length, petal length and petal width measurements and yet the largest sepal widths.
None of the species individual averages are close to the overall average for all species.
The difference between species seems much more pronounced for petal measurements than sepal measurements 

## Standard Deviation
How close measurements are the to average - how wide is the bell curve? Is there even a bell curve? Is the data normally distributed?


## Visualising the Data
Data visualisation helps us to identify interesting aspects of the data in a more clean and efficient manner than just looking at numbers.

Univariate plots help us to understand each individual attribute (e.g. histograms, box plots while multivariate allows us to visualise the relationships between the attributes (e.g. scatter plots). https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

## Pairplots
A look at the pairplot below shows that iris setosa (shown in blue) appears visually to be quite separate from the other two species. While there is quite a bit of observable overlap between versicolor and virginica, particularly in terms of setosa measurements, setosa appears to the significantly distinct.
![Pairplot](Images/pairplot.PNG) 

### Scatterplots
The scatterplots show how sepal and petal lengths and widths relate to one another. 

The scatterplots show how one species of iris, the Setosa, is almost entirely distinct from the other two species. A straight line could be drawn to separate the sepal and petal measurements of Setosa from those of the other species, and this difference is particularly evident with petal measurements. 

If one sees an iris with short, narrow petals and short but wide sepals, it could be reliably predicted that the particular species is Setosa.  

Versicolor and Virginica are not very distinguishable from one another in terms of sepal measurements but looking at the petal data, Virginica irises seem more likely to have longer, wider petals than Versicolor.

![Petal Measurements by Species](Images/scatter.petal.png)
![Sepal Measurements by Species](Images/scatter.sepal.png)


The histograms and scatterplots shown in the pairplot, allow us to se at a glance how separable the setosa data is from the other two iris species across all measurements. 
While iris versicolor and virginica are not as distinct from one another, the pairplots show that they are more separable by petal measurements than sepal measurements. 


## Correlations
Correlation figures demonstrate that petal length and petal width are both highly correlated with sepal length
Petal length and width are very highly correlated
However, sepal width is not highly correlated with any other measurement. The smallest correlation is between sepal length and width, suggesting that they can be vastly different. When we were previously looking at averages, we saw that the species setosa has the smallest average measurements in petal length and width and in sepal length and yet had the largest average sepal width. 
?Look at particular correlations between species to see if setosa is affecting these figures. 

## 3. Machine Learning
A machine learning program learns from information provided by previous examples. In the case of the Iris Dataset, this would be supervised learning as we a
The patterns identified in the iris dataset can be used to create a predictive algorithm to determine a species of iris flower based on sepal and petal measurements.

https://www.youtube.com/watch?v=hd1W4CyPX58&t=187s
It is a supervised learning problem as we are presented with both input (iris measurements) and output (iris species) pairs. The information from these pairings should ideally allow us to accurately predict a species of iris when presented with new data inputs. The iris dataset has become popular in machine learning teaching due to the strong link between species and measurements, particularly in the case of iris setosa.

There are several algorithms available in the sklearn Python library that can be used to build a machine learning model for the Iris Dataset including:
- Logistic Regression
- Linear Discriminant Analysis
- K-Nearest Neighbors
- Classification and Regression Trees 
- Gaussian Naive Baynes
- Support Vector Models

These models have been used and tested for accuracy with the iris dataset several times  https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
here: https://www.kaggle.com/kamrankausar/iris-dataset-ml-and-deep-learning-from-scratch
and here: https://medium.com/codebagng/basic-analysis-of-the-iris-data-set-using-python-2995618a6342

For the purposes of the current investigation, I will be using a K-Nearest Neighbors model to illustrate how one may go about constructing a machine learning model.

https://www.youtube.com/watch?v=cKxRvEZd3Mw
The steps to machine learning are:
1. Collect training data
2. Train the classifier
3. Make predictions
Iris measurements = features

For machine learning, one must split the dataset into training data and test data
Swain et al (2012) used 75 for training and 75 for testing
xxx https://www.kaggle.com/sharmajayesh76/iris-data-train-test-split also halves the data
xxx https://machinelearningmastery.com/machine-learning-in-python-step-by-step/ 80% training, 20% testing
xxx https://python-guide-kr.readthedocs.io/ko/latest/scenarios/ml.html 140 for training, 10 for testing (90%, 10%)
xxx https://www.kaggle.com/kamrankausar/iris-dataset-ml-and-deep-learning-from-scratch 70% training, 30% testing
xxx https://medium.com/codebagng/basic-analysis-of-the-iris-data-set-using-python-2995618a6342 80% training, 20% testing
https://github.com/justmarkham/scikit-learn-videos/blob/master/04_model_training.ipynb


Not sure if I should go into the Machine Learning side of things
Requires scikit knowledge
This is a nice intro https://python-guide-kr.readthedocs.io/ko/latest/scenarios/ml.html
More complex https://www.kaggle.com/sharmajayesh76/iris-data-train-test-split
https://www.ritchieng.com/machine-learning-iris-dataset/
https://www.kaggle.com/kamrankausar/iris-dataset-ml-and-deep-learning-from-scratch

## K Nearest Neighbor (kNN) 
KNN can be used for classification prediction models. A model can be built by kNN using a dataset that contains input features and output labels

and, when presented with new data, uses Euclidean Distance to measure the distance between the new data points and k number of established data points.(can be adjusted based on the size of the dataset)  If the majority of data point nearest the new data point match a particular species, the model will ascertain that the new measurements presented belong to a particular species.

There is no ideal value for k that would make the model most accurate. It is advisable to try different values for k to see what returns the most accurate predictions.
https://discuss.analyticsvidhya.com/t/how-to-choose-the-value-of-k-in-knn-algorithm/2606/13

In the table below, I have tested 10 different k values 10 different times and chose the k value of 13 based on the mean accuracy percentage.
![Choosing the k value](Images/k_values.PNG)


The image below demonstrates KNN classification that checks two different instances. In the first instance (K = 3) the model would checks the 3 nearest neighbours and determine that the new data point belongs to Class B. However, if the KNN is expanded (K = 7), allowing the model to check the 7 nearest neighbours, the new datapoint is likely to belong to Class A.
![KNN Classification](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1531424125/Knn_k1_z96jba.png)

kNN is useful as an introduction to machine learning models but it does not strictly speaking actually create a model that is trained to interpret data. It runs through the dataset for each and every prediction meaning it consumes a lot of time and space. https://stackoverflow.com/questions/10814731/knn-training-testing-and-validation


## Exploring the Data
Python Libraries
- pandas
- matplotlib
- seaborn
- numpy
- sklearn
Visual Studio Code
Jupyter Notebook - for clean image output


While pandas was used to describe and explore the dataset, it needs to be expressed as a numpy array in order to demonstrate a machine learning algorithm.

## References
Bezdek, J. C., Keller, J. M., Krishnapuram, R., Kuncheva, L. I., & Pal, N. R.       (1999) *Correspondence: Will the Real Iris Data Please Stand Up?*. IEEE        Transactions on Fuzzy Systems, 7: 3, June 1999.

Fisher, R. A. (1936) *The Use of Multiple Measurements in Taxonomic Problems.*      Annals of Eugenics, 7.2. 

Swain, M., Dash, S. K., Dash, S., & Mohapatra, A. (2012) *An Approach for Iris     Plant Classification Using Neural Network*. International Journal on Soft       Computing, 3: 1, February 2012. 


# Notes

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




Look at the dataset in terms of: 
* Classification - organising data by categories
* Linear discriminant analysis 
* Clustering
* Data visualisation - graphs
* Pattern recognition - species could be predicted based on data measurements
* Machine learning such as neural networks - predictive nature of the data 