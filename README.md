# MA707 Machine Learning Final Project 
- __Author:__ Rajasekharsingh Bondili, Rachel Hu, Madhurya Baruah
- __Class:__ MA707 Machine Learning
- __Term:__ 2019 Spring

## Introduction 
The main target of this report is to investigate different statistical models and to determine the features, transformers, estimators and hyperparameters to create the best predictive model for an given dataset. The datasets are collected from different commodity index exchanges across the globe as well as incorporated the news from several mining and finance news sources. The techniques we used here are feature engineering, feature selection and machine learning modeling. The goal is to predict the 5 time charter price `(bci_5tc)` based on the time series and text mining features we have created from the datasets.


## 1. Dataset description
The datasets used in the project contain indices in terms of dollars or sdrs, indices of market prices for non-fuel commodities such as metals, energy, livestock and meat and agricultural products and average weekly prices for these commodities from 2011 till 2019. The datasets are collected from different exchanges such as **Baltic Exchange Dry Index (BDI), Intercontinental Exchange (ICE), The Rogers International Commodity Index® (“RICI®”)**. 

The `5tc_plus_ind_vars.csv` dataset also contains the 5 day time charter price as well. A time charter is the hiring of a vessel for a specific period of time; the owner still manages the vessel but the charterer selects the ports and directs the vessel where to go. The charterer pays for all fuel the vessel consumes, port charges, commissions, and a daily hire to the owner of the vessel.

Two other datasets are also used which have the content from different news sources accross the world such as ***mining.com, bloomberg news reuters, mining equipment and supplier news etc***. They contain the tags, titles, and contents from all the different sources from 04/01/2008 to 01/15/2019.

The Baltic Exchange Capesize Index (BCI) is a daily average calculated from the reports of an independent international board of Panellists. These Panellists are required to make a daily assessment on a basket of timecharter and voyage routes in the dry bulk shipping market representative of Capesize vessels. 

Basic economic principles of supply and demand typically drive the commodities markets: lower supply drives up demand, which equals higher prices, and vice versa. Major disruptions in supply, such as a widespread health scare among cattle, might lead to a spike in the generally stable and predictable demand for livestock. On the demand side, global economic development and technological advances often have a less dramatic, but important effect on prices. Case in point: The emergence of China and India as significant manufacturing players has contributed to the declining availability of industrial metals, such as steel, for the rest of the world.

The different types of Commodities fall into the following four categories:

  - Metals (such as iron, gold, silver, platinum and copper)
  - Energy (such as coal, crude oil, heating oil, natural gas and gasoline)
  - Livestock and Meat (including lean hogs, pork bellies, live cattle and feeder cattle)
  - Agricultural (including corn, soybeans, wheat, rice, cocoa, coffee, cotton and sugar)
  

## 2. Plan
In order to select the best model to predict the defined target variable, we follow the next steps :
- Data cleaning: All the three datasets will be cleaned and joined, then used to train and test the models individually. The plan is to combine the time series data `5tc_plus_ind_vars.csv` with the mining datasets individually to create two merged datasets.
- Class: Different wrapper classes are defined to perform individual pre-processing on the datasets before fitting into another 
- Estimator and pipelines: Several estimators are to be used seperately as well as combined in pipelines. 
- Feature selection and extraction: Different scikit learn techniques will be used for feature selection and extraction from the datasets. Since the mining datasets are all text contents, these texts has to be converted to numerical data to be fitted into the machine learning model hence feature creation/extraction methods are used.
- Model design and hyperparameter tuning: Different regression models are defined and then trained with separate training datasets to get their test scores. These models are then tested with several hyperparameters and the scores compared using GridsearchCV.

Finally the best fit models will be selected with their tuned hyperparameters and then the test dataset is used to get the prediction score and other score metrics for the best model.
