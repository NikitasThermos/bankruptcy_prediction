# Bankruptcy Prediction
A machine learning project for predicting bankruptcy in the context of 'Data Mining' university course
<h3>Tools</h3>

[![Tools](https://skillicons.dev/icons?i=py,sklearn,tensorflow)](https://skillicons.dev) 

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Results](#results)
4. [Dataset Features](#features)

<a name="introduction"></a>
## Introduction
The goal of the project was to develop ML models ob a dataset that has irregularities and anomalies. For taht reason the project is splited in two main parts. The first part includes the preprocessing of the data and the devlopment of a pipeline that it can transform the data initially to a state that is required from the ML models and secondly to increase the information that we can obtain from the original data for better perfomance. For the second part of the project we implement few ML models and compare the results on a test dataset.

<a name="dataset"></a>
## Dataset
The project works with a subset of " Tomczak,Sebastian. (2016). Polish companies bankruptcy data. UCI Machine Learning Repository." which is available [here](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data). The dataset contains financial information about Polish companies analyzed in the period 2000-2012 and the data were collected from Emerging Markets Information Service(EMIS). Specifically, for each company the dataset contains 64 annual financial statistics and also a label that indicates bankruptcy status after certain amount of years. You can find what each feature of the dataset represents on the [Dataset Feeatures](#features) section 

The subset contains data collected only from the first year of the forecasting period and the label indicates the bankruptcy status after 5 years.  


<a name="results"></a>
## Results

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
