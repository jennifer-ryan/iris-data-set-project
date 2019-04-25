# Iris Dataset Project 2019
This repository contains an explanation and exploration of the famous Iris Dataset as part of the assessment in the  Programming and Scripting module for the Higher Diploma on Data Analytics with Galway-Mayo Institute of Technology.
 
## 1. About this Repository
The repository is made up of the following files and folders:
* A **README** file that contains descriptions of the Iris Dataset, exploratory data analysis using statistics and visualisation, and a very basic illustration of how the dataset can be used in machine learning.
* The **iris.csv** file which contains the complete dataset downloaded from [here](https://gist.github.com/curran/a08a1080b88344b0c8a7#file-iris-csv-L1). Note that there have been slight discrepancies observed in some electronic versions of the dataset with small variations observed when compared to the original dataset that was published in 1936 (Bezdek et al, 1999) but these differences are not substantial enough to have any major effect on the overall patterns in the dataset.
* Two **Jupyter Notebooks** created using Python: 
    * **Iris Dataset Exploratory Data Analysis** uses the *pandas* library for statistical investigations and *matplotlib* and *seaborn* for data visualisation.
    * **Machine Learning** that interprets the dataset as *numpy* arrays and builds a basic model using *sklearn*, specifically the *K-Nearest Neighbor* algorithm. 
* **Images** folder that contains .png files of some of the data visualisation performed in the Exploratory Data Analysis notebook that are embedded for description purposes in this README. 

## 2. Python Coding Methodology
As a novice in the Python language, this project challenged me to become familiar with several new libraries that have been widely used to investigate the dataset. For the exploratory portion of the project, I learned the basics of using the pandas library to read datasets ([Mester, 2019](https://data36.com/pandas-tutorial-1-basics-reading-data-files-dataframes-data-selection/), [Pandas-Docs](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) as it is an excellent tool for data manipulation and statistics that is relatively easy to learn. To create graphs, I used a mixture of matplotlib and seaborn, which was developed based on matplotlib and creates more attractive graphs ([Mendis, 2019](https://www.kdnuggets.com/2019/04/data-visualization-python-matplotlib-seaborn.html)).

The machine learning portion of the project uses scikit-learn which is built to interpret a dataset as a numpy array rather than through pandas. Scikit-learn is a very powerful library and is quite complex so for the purposes of this investigation, I attempted to create a very basic illustration of a potential machine learning programming using the K-Nearest Neighbors algorithm. 

Rather than presenting the code generated for this project as a series of .py files, I decided to learn how to use a Jupyter Notebook for code presentation as output is presented in a much cleaner fashion and is altogether more legible.

## 3. The Dataset
The Iris Dataset consists of 50 samples each of three different species of iris flower: *setosa*, *versicolor* and *virginica*. It contains four different measurements for each sample in centimetres - the length and width of sepals and petals - making it a multivariate dataset.

![Iris Species](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png)

The data was collected by botanist Edgar Anderson in the Gaspé Peninsula and popularised when it was used by biologist and statistician Ronald Fisher in his 1936 paper *The Use of Multiple Measurements in Taxonomic Problems* to demonstrate how statistics can be used for classification. He argues that, based on some significant attribute differences between the species in this dataset, iris group membership could potentially be determined by sepal and petal measurements alone - a method that would become known as linear discriminant analysis - and that new iris flowers could be classified based on the statistical information gleaned from the dataset. 

The Iris Dataset remains a commonly used example as an introduction to exploratory data analysis, pattern recognition, and machine learning algorithms for the following reasons:
* It is a complete, balanced dataset in that there are no null values and each class is equally represented. 
* Each of the four features (sepal and petal length and width) are measured in the same units (centimetres).
* One iris species (setosa) is linearly separable from the other two. While the other species have some overlap, they are still largely distinguishable from one another in some measurements. Thus, classification is relatively easy and, by extension, the predictive capability of the data is quite strong. 

## 4. Exploratory Data Analysis
*To be read in conjunction with Jupyter Notebook entitled **Iris Dataset Exploratory Analysis***

The following resources were used to develop a familiarity with the pandas library and previous exploratory analysis of the dataset:
https://medium.com/@harimittapalli/exploratory-data-analysis-iris-dataset-9920ea439a3e
https://www.kaggle.com/lalitharajesh/iris-dataset-exploratory-data-analysis
https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

Exploratory Data Analysis allows us to better understand the data through statistical and visual techniques in order to form hypotheses and uncover potential patterns in the data. 
 
Generally, the first things to look at when confronted with a new dataset are the structure of the dataset and basic information about its contents (Refs). Pandas allows us to see that the dataset is comprised of 150 rows and 5 columns; 4 of these columns are float datatypes containing the measurements and the last one is an object datatype that contains the species names. There are no null values in the dataset that need to be accounted for in later analysis. We can see that the dataset is well balanced with each species accounting for 50 samples.
![Iris Dataset Information](Images/iris.info.PNG) ![Iris Species Distribution](Inages/iris.species.PNG)




## Averages and Standard Deviations
Simply observing the averages of each measurement by species seems to demonstrate significant differences between the species. For instance, setosa has the smallest sepal length, petal length and petal width measurements and yet the largest sepal widths.
None of the species individual averages are close to the overall average for all species.
The difference between species seems much more pronounced for petal measurements than sepal measurements 
![Means: Total and by Species](Images/iris.means.PNG)

Standard Deviation
How close measurements are the to average - how wide is the bell curve? Is there even a bell curve? Is the data normally distributed?


## Correlations
Correlation figures demonstrate that petal length and petal width are both highly correlated with sepal length
Petal length and width are very highly correlated
However, sepal width is not highly correlated with any other measurement. The smallest correlation is between sepal length and width, suggesting that they can be vastly different. When we were previously looking at averages, we saw that the species setosa has the smallest average measurements in petal length and width and in sepal length and yet had the largest average sepal width. 
?Look at particular correlations between species to see if setosa is affecting these figures. 

## Visualising the Data
Data visualisation helps us to identify interesting aspects of the data in a more clean and efficient manner than just looking at numbers.

Univariate plots help us to understand each individual attribute (e.g. histograms, box plots while multivariate allows us to visualise the relationships between the attributes (e.g. scatter plots). https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

## Pairplots
A look at the pairplot below shows that iris setosa (shown in blue) appears visually to be quite separate from the other two species. While there is quite a bit of observable overlap between versicolor and virginica, particularly in terms of sepal measurements, setosa appears to the significantly distinct. Petal length and width in the setosa are significantly smaller than either versicolor or virginica.
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


## 5. Machine Learning
The following resources were used to develop a familiarity with the sklearn library and previous machine learning examples utilising the Iris Dataset: 
https://github.com/justmarkham/scikit-learn-videos/blob/master/04_model_training.ipynb
https://www.youtube.com/watch?v=hd1W4CyPX58&t=187s
https://www.youtube.com/watch?v=kzjDUr-7uRw


A machine learning program learns from data provided by previous examples. In the case of the Iris Dataset, this would be supervised learning as we a
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

To create a machine learning program, the dataset is often split so that a certain percentage is used to train the program and the rest is used to test the program. There is no correct training/testing ratio but generally a [70/30 split] (https://www.researchgate.net/post/Is_there_an_ideal_ratio_between_a_training_set_and_validation_set_Which_trade-off_would_you_suggest) is adopted (). Other examples I have come across use a 50/50 split (Swain et al, 2012; Sharma, 2017), 80/20 split (Brownlee, 2016; Ogundowole, 2017)

For machine learning, one must split the dataset into training data and test data
Swain et al (2012) used 75 for training and 75 for testing
xxx https://www.kaggle.com/sharmajayesh76/iris-data-train-test-split also halves the data
xxx https://machinelearningmastery.com/machine-learning-in-python-step-by-step/ 80% training, 20% testing
xxx https://python-guide-kr.readthedocs.io/ko/latest/scenarios/ml.html 140 for training, 10 for testing (90%, 10%)
xxx https://www.kaggle.com/kamrankausar/iris-dataset-ml-and-deep-learning-from-scratch 70% training, 30% testing
xxx https://medium.com/codebagng/basic-analysis-of-the-iris-data-set-using-python-2995618a6342 80% training, 20% testing
https://github.com/justmarkham/scikit-learn-videos/blob/master/04_model_training.ipynb


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
![KNN Classification](https://pbs.twimg.com/media/DmVRIqrXcAAOvtH.jpg)

kNN is useful as an introduction to machine learning models but it does not strictly speaking actually create a model that is trained to interpret data. It runs through the dataset for each and every prediction meaning it consumes a lot of time and space. https://stackoverflow.com/questions/10814731/knn-training-testing-and-validation


While pandas was used to describe and explore the dataset, it needs to be expressed as a numpy array in order to demonstrate a machine learning algorithm.

## 6. References
Bezdek, J. C., Keller, J. M., Krishnapuram, R., Kuncheva, L. I., & Pal, N. R. (1999) *Correspondence: Will the Real Iris Data Please Stand Up?*. IEEE Transactions on Fuzzy Systems, 7: 3, June 1999.

Brownlee, J. (2016) *Your First Machine Learning Project in Python Step-by-Step.* https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

Codesbay (2018). *Machine Learning 'Hello World' with Scikit Learn: Chapter 5: Using the Iris Dataset.* YouTube. https://www.youtube.com/watch?v=kzjDUr-7uRw

Fisher, R. A. (1936) *The Use of Multiple Measurements in Taxonomic Problems.* Annals of Eugenics, 7.2. 

Kadam, A. (2017) *Iris Data Analysis*. GitHub Repository https://github.com/ashKadam/IrisDataAnalysis/blob/master/Iris.py

Mendis, A. (2019) *Data Visualization in Python: Matplotlib vs Seaborn.* KDnuggets. https://www.kdnuggets.com/2019/04/data-visualization-python-matplotlib-seaborn.html

Mester, T. (2018). Pandas Tutorial Series. [Part 1](https://data36.com/pandas-tutorial-1-basics-reading-data-files-dataframes-data-selection/), [Part 2](https://data36.com/pandas-tutorial-2-aggregation-and-grouping/), [Part 3](https://data36.com/pandas-tutorial-3-important-data-formatting-methods-merge-sort-reset_index-fillna/).

Mittapalli, H. (2018) *Exploratory Data Analysis.* Medium. https://medium.com/@harimittapalli/exploratory-data-analysis-iris-dataset-9920ea439a3e

Ogundowole, O. O. (2017) *Basic Analysis of the Iris Dataset Using Python.* Medium. https://medium.com/codebagng/basic-analysis-of-the-iris-data-set-using-python-2995618a6342

Pandas-Docs (2019) *10 Minutes to Pandas.* https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html

Rajesh, L. (2018) *Iris Dataset - Exploratory Data Analysis.* Kaggle Notebook. https://www.kaggle.com/lalitharajesh/iris-dataset-exploratory-data-analysis

Sharma, J. (2017) *Iris Data Train_Test_Split.* Kaggle Notebook. https://www.kaggle.com/sharmajayesh76/iris-data-train-test-split

Swain, M., Dash, S. K., Dash, S., & Mohapatra, A. (2012) *An Approach for Iris Plant Classification Using Neural Network*. International Journal on Soft Computing, 3: 1, February 2012. 


# Notes

Brownlee (2016):
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