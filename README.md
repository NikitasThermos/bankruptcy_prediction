# Bankruptcy Prediction
A machine learning project for predicting bankruptcy in the context of 'Data Mining' university course
<h3>Tools</h3>

[![Tools](https://skillicons.dev/icons?i=py,sklearn,tensorflow)](https://skillicons.dev) 

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Processing](#processing)
4. [Results](#results)
5. [Further Improvements](#improvements)
6. [Deployment](#deployment)
7. [Dataset Features](#features)

<a name="introduction"></a>
## Introduction
The goal of the project was to develop ML models with a dataset that has irregularities and anomalies. For that reason the project is splited in two main parts. The first part includes the preprocessing of the data and the devlopment of a pipeline that it can transform the data initially to a state that is required from the ML models and secondly to increase the information that we can obtain from the original data for better perfomance. For the second part of the project we implement few ML models and compare the results on a test dataset.

<a name="dataset"></a>
## Dataset
The project works with a subset of " Tomczak,Sebastian. (2016). Polish companies bankruptcy data. UCI Machine Learning Repository." which is available [here](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data). The dataset contains financial information about Polish companies analyzed in the period 2000-2012 and the data were collected from Emerging Markets Information Service (EMIS). Specifically, for each company the dataset contains 64 annual financial statistics and also a label that indicates bankruptcy status after certain amount of years. You can find what each feature of the dataset represents on the [Dataset Feeatures](#features) section. 

The subset that we worked with contains data collected only from the first year of the forecasting period and the labels indicate the bankruptcy status after 5 years. Therfore we can distinguish the observations into two classes that we can simply refer to them as negative and positive classes. The positive class contains the observavions which have a label equal to one indicating bankruptcy after 5 years while the rest of the observations are part of the negative class. The ratio of observations for each class in the training and the test set are presented below.

| Set |  Negative Observations | Positive Observations | 
| --- | --- | --- |
| Training | 6756 | 271 |  
| Test |  3486 | 514 | 

Thus the problem can be thought as a binary classification task with the goal of each model being to correctly identify the label of each observation. While working with that dataset we came across some irregularities that either restricted the input of the data on most ML models or reduced the perfomance of the models. Some problems of the dataset include the following:

* Missing Values
* Outlier instances/values
* Duplicate instances
* Imbalanced classes
* Small amount of training instances and features

<a name="processing"></a>
## Data Processing
The first step of preprocessing is the removal of duplicate observations. We believe that it is highly unlikely that two observations would have the same values across 64 features and thus these observations have been mistakenly added twice. Further all of the duplicate observations are part of the majority class and by removing them we improve the balance between the two classes. In total, 82 observations were removed in that process. 

By examining the statistics of the dataset we noticed that for many features the minimum and maximum values are way too far from the mean values. Furthermore, these extreme values are not parts of the same instances so it is not a simply case of outlier observations. We procedeed by removing these value resulting in a great improvement of perfomance for all the models. The removal of the outlier values is achieved by obtaining a standardize version of the dataset and removing the values which are greater than a threshold.

The missing values that exist in the dataset are filled with an Imputer. We use the KNNImputer that fills the missing values based on the closest neighbors of each instance. Also, most of the ML algorithms perform better with scaled data so each feature has a similar range. We use the RobustScaler which is not influenced by outliers. A comparison of different scalers in a dataset with outliers can be found [here](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html).

Due to the small amount of features that are included in the original dataset we also included the addition of new features, hoping to provide more usefull information for our models. To achieve that, a common way is the PolynomialFeatures class that adds higher degrees and combinations of the original features. 

Finally, we tried to deal with the problem of the imbalanced classes in two different ways: 

1. Class Weights 

Most ML models give the option of providing a weight for each of the classes. With that we can specify how much the model should focus on the observations of each class. Giving a large weight on the minority class means that each observation of that class will have a greater impact on the update of model's parameters thus balancing the training of the model. 

2. Sampling

With sampling we can generate more instances of a class so the model will have more observations to work with. There are many ways to add new observations based on the original ones but we chose the ADASYN sampler from Imbalanced-Learn. [Here](https://imbalanced-learn.org/stable/over_sampling.html#smote-adasyn) is a comparison of the samplers that Imbalanced-Learn provides.

Over-sampling of the minority class provides a better perfomance for all the models compared to class weights so we chose that for our preprocessing.


<a name="results"></a>
## Results
Because the dataset is heavily inbalanced between the two classes, using the accuracy of the models' predictions as the main perfomance measure is not optimal. For example a model that only predicts every observation as the majority class. which is the negative class in our case, will have a 96% accuracy on this training set. For that reason we can use the Precision and Recall measures that indicate how accurate is the model when giving a positive prediction and how many positive observations can identify. To calculate these measures we need to categorize each prediction of a model based on the following: 
* True Negative (TN): A negative observation that is predicted as negative
* False Negative (FN):  A positive observation that is predicted as negative
* True Positive (TP): A positive observation that is predicted as positive
* False Positive (FP): A negative observation that is predicted as positive

Using the above we can calculate the Precision and Recall of each model: 

Precision = $\frac{TP}{TP + FP}$

Recall = $\frac{TP}{TP + FN}$

Finally we can combine these two to get the F1 score which is the harmonic mean of Precision and Recall. Based on these measures the perfomance of each moel is presented in the table below:

| Model | Precision | Recall | F1 | 
| --- | --- | --- | --- | 
| Linear Logistic Loss | 0.28 | 0.59 | 0.38
| Support Vector Machine | 0.28 | 0.59 | 0.38
| Support Vector Machine - Polynomial | 0.27 | 0.73 | 0.40
| Support Vector Machine - RBF | 0.27 | 0.70 | 0.39
| Random Forest | 0.63 | 0.44 | 0.52 
| Dense Neural Network | 0.56 | 0.52 | 0.54

<a name="improvements"></a>
## Further Improvements

One of the areas that provided a great perfomance improvement was the preprocessing of the data. Each time a new addition was made in the preprocessing procedure, such as removing duplicate observations, removing extreme values or adding new features, a great improvement was achieved with the same models. Further additions to preprocessing could be enough for better results. For example, we have not tried dropping the features with high percentage of missing values and also extracting the most important features with a random forest and training only on that subset of features. 

Also at this momment there is not hyperparameter tuning for the DNN, that could provide better results. Given that we used TensorFlow for the DNN implementation an easy way to achieve that would be Keras Tuner. Additionaly, we did not run cross-validation for DNN. With cross-validation, the model can use more training instances that will potentially improve the perfomance and will be validated on more observations that can improve the generalization of the model. 

Finally, using an ensemble of the best models can provide better results. For example, a combination of Random Forest's and DNN's predictions can potentially give better final results.


<a name="deployment"></a> 
## Deployment 

You can clone the project locally with the following: 
```bash
git clone https://github.com/NikitasThermos/bankruptcy_prediction
```
Then there are two main ways to run it: 
1. Jupyter Notebooks

The whole preprocessing of the data and the models implementation is available at a notebook file [here](#bankruptcy-prediction.ipynb). You can run the notebook on Google Colab if you want to use additional computing resources. 

2. Python Modules



<a name="features"></a >
## Dataset Features

| Feature | Ratio |
| :--- | :--- |
| X1 |	net profit / total assets 
| X2 |	total liabilities / total assets
| X3 |	working capital / total assets
| X4 |	current assets / short-term liabilities
| X5 |	[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365
| X6 |	retained earnings / total assets
| X7 |  EBIT / total assets
| X8 |	book value of equity / total liabilities
| X9 |	sales / total assets
| X10 |	equity / total assets
| X11 |	(gross profit + extraordinary items + financial expenses) / total assets
| X12 |	gross profit / short-term liabilities
| X13 |	(gross profit + depreciation) / sales
| X14 |	(gross profit + interest) / total assets
| X15 |	(total liabilities * 365) / (gross profit + depreciation)
| X16 |	(gross profit + depreciation) / total liabilities
| X17 |	total assets / total liabilities
| X18 |	gross profit / total assets
| X19 |	gross profit / sales
| X20 |	(inventory * 365) / sales
| X21 |	sales (n) / sales (n-1)
| X22 |	profit on operating activities / total assets
| X23 |	net profit / sales
| X24 |	gross profit (in 3 years) / total assets
| X25 |	(equity - share capital) / total assets
| X26 |	(net profit + depreciation) / total liabilities
| X27 |	profit on operating activities / financial expenses
| X28 |	working capital / fixed assets
| X29 |	logarithm of total assets
| X30 |	(total liabilities - cash) / sales
| X31 |	(gross profit + interest) / sales
| X32 |	(current liabilities * 365) / cost of products sold
| X33 |	operating expenses / short-term liabilities
| X34 |	operating expenses / total liabilities
| X35 |	profit on sales / total assets
| X36 |	total sales / total assets
| X37 |	(current assets - inventories) / long-term liabilities
| X38 |	constant capital / total assets
| X39 |	profit on sales / sales
| X40 |	(current assets - inventory - receivables) / short-term liabilities
| X41 |	total liabilities / ((profit on operating activities + depreciation) * (12/365))
| X42 |	profit on operating activities / sales
| X43 |	rotation receivables + inventory turnover in days
| X44 |	(receivables * 365) / sales
| X45 |	net profit / inventory
| X46 |	(current assets - inventory) / short-term liabilities
| X47 |	(inventory * 365) / cost of products sold
| X48 |	EBITDA (profit on operating activities - depreciation) / total assets
| X49 |	EBITDA (profit on operating activities - depreciation) / sales
| X50 |	current assets / total liabilities
| X51 |	short-term liabilities / total assets
| X52 |	(short-term liabilities * 365) / cost of products sold)
| X53 |	equity / fixed assets
| X54 |	constant capital / fixed assets
| X55 |	working capital
| X56 |	(sales - cost of products sold) / sales
| X57 |	(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)
| X58 |	total costs /total sales
| X59 |	long-term liabilities / equity
| X60 |	sales / inventory
| X61 |	sales / receivables
| X62 |	(short-term liabilities *365) / sales
| X63 |	sales / short-term liabilities
| X64 |	sales / fixed assets
