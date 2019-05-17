# Databricks notebook source
# MAGIC %md ##MA707 Report - Introduction (spring 2019, DataHeroes)
# MAGIC - __Author:__ Rajasekharsingh Bondili, Rachel Hu, Madhurya Baruah
# MAGIC - __Class:__ MA707 Machine Learning
# MAGIC - __Term:__ 2019 Spring

# COMMAND ----------

# MAGIC %md ## Introduction 
# MAGIC 
# MAGIC This report is to understand the basic structure of machine learning models and use these tools and methods to design a prediction model for dry commodities based on the data collected from different commodity index exchanges across the globe as well as incorporate the news and contents from several mining and finance news sources. The data provided consist of both numerical datas as well text data sets, hence these datas need be cleaned and converted to single format to be fed into the machine learning models.

# COMMAND ----------

# MAGIC %md ## Links to report sections
# MAGIC 1. [Demonstrate classes](https://bentley.cloud.databricks.com/#notebook/1607119/command/1607120): demonstrate individual wrapper classes
# MAGIC 1. [Preprocessing pipelines](https://bentley.cloud.databricks.com/#notebook/1607264/command/1607265): demonstrate preprocessing pipelines
# MAGIC 2. [Estimator pipelines](https://bentley.cloud.databricks.com/#notebook/1607450/command/1607451): demonstrate estimator pipelines
# MAGIC 3. [Investigations](https://bentley.cloud.databricks.com/#notebook/1607697/command/1607698): use `GridSearchCV` with estimator pipelines and feature-target datasets (created with preprocessing pipelines) to learn which hyper-parameters and which features work well and which don't
# MAGIC 
# MAGIC Links to other included notebooks 
# MAGIC 1. [Raw dataset (inc)](https://bentley.cloud.databricks.com/#notebook/1607016/command/1607030)
# MAGIC 1. [Feature creation (inc)](https://bentley.cloud.databricks.com/#notebook/1607061/command/1607062)
# MAGIC 2. [Feature selection (inc)](https://bentley.cloud.databricks.com/#notebook/1607074/command/1607075)
# MAGIC 1. [Estimators (inc)](https://bentley.cloud.databricks.com/#notebook/1607089/command/1607090)
# MAGIC 1. [Pipeline functions (inc)](https://bentley.cloud.databricks.com/#notebook/1608201/command/1608202)

# COMMAND ----------

# MAGIC %md ## Contents
# MAGIC 1. Dataset description
# MAGIC 1. Objectives
# MAGIC 1. Plan

# COMMAND ----------

# MAGIC %md ## 1. Dataset description

# COMMAND ----------

# MAGIC %md 
# MAGIC The dataset used in the project contains indices in terms of dollars or sdrs, indices of market prices for non-fuel commodities such as metals, energy, livestock and meat and agricultural products and average weekly prices for these commodities from 2011 till 2019. It contains data from different exchanges such as **Baltic Exchange Dry Index (BDI), Intercontinental Exchange (ICE), The Rogers International Commodity Index® (“RICI®”)**. 
# MAGIC 
# MAGIC The `5tc_plus_ind_vars.csv` dataset also contains the 5 day time charter price as well. A time charter is the hiring of a vessel for a specific period of time; the owner still manages the vessel but the charterer selects the ports and directs the vessel where to go. The charterer pays for all fuel the vessel consumes, port charges, commissions, and a daily hire to the owner of the vessel.
# MAGIC 
# MAGIC Two other datasets are also used which contains the content from different news sources for iron and coal accross the world such as ***mining.com, bloomberg news reuters, mining equipment and supplier news etc***. They contain the tags, title, and content from all the different sources from 04/01/2008 to 01/15/2019.
# MAGIC 
# MAGIC The Baltic Exchange Capesize Index (BCI) is a daily average calculated from the reports of an independent international board of Panellists. These Panellists are required to make a daily assessment on a basket of timecharter and voyage routes in the dry bulk shipping market representative of Capesize vessels. 
# MAGIC 
# MAGIC Basic economic principles of supply and demand typically drive the commodities markets: lower supply drives up demand, which equals higher prices, and vice versa. Major disruptions in supply, such as a widespread health scare among cattle, might lead to a spike in the generally stable and predictable demand for livestock. On the demand side, global economic development and technological advances often have a less dramatic, but important effect on prices. Case in point: The emergence of China and India as significant manufacturing players has contributed to the declining availability of industrial metals, such as steel, for the rest of the world.
# MAGIC 
# MAGIC The different types of Commodities fall into the following four categories:
# MAGIC 
# MAGIC   - Metals (such as iron, gold, silver, platinum and copper)
# MAGIC   - Energy (such as coal, crude oil, heating oil, natural gas and gasoline)
# MAGIC   - Livestock and Meat (including lean hogs, pork bellies, live cattle and feeder cattle)
# MAGIC   - Agricultural (including corn, soybeans, wheat, rice, cocoa, coffee, cotton and sugar)

# COMMAND ----------

# MAGIC %md ## 2. Objectives

# COMMAND ----------

# MAGIC %md 
# MAGIC The objective of this notebook is to investigate different statistical models and to determine the features, transformers, estimateors and hyperparameters to create the best predictive model for an given dataset. 
# MAGIC Using the commodity prices, tags and contents from different news sources for coal and iron ore, the goal is to predict the 5 time charter price `(bci_5tc)`. 

# COMMAND ----------

# MAGIC %md ## 3. Plan

# COMMAND ----------

# MAGIC %md Inorder to design the best fit model for predicting the defined target variable the main steps explained in this report are:
# MAGIC - Data cleaning: All the three datasets will be cleaned and manipulated so that there is no missing values and then these three different datasets will be combined together to get different merged data which can be used to train and test the models individually. The plan is to combine the time series data `5tc_plus_ind_vars.csv` with the mining datasets individually to create two merged datasets.
# MAGIC - Class: Different wrapper classes are defined to perform individual pre-processing on the datasets before fitting into another 
# MAGIC - Estimator and pipelines: Several estimators are to be used individually as well as combined in pipelines. 
# MAGIC - Feature selection and extraction: Different scikit learn techniques will be used for feature selection and extraction from the datasets. Since the mining datasets are all text contents, these texts has to be converted to numerical data to be fitted into the machine learning model hence feature creation/extraction methods are used.
# MAGIC - Model design and hyperparameter tuning: Different regression models are defined and then trained with separate training datasets to get their test scores. These models are then tested with several hyperparameters and the scores compared using GridsearchCV.
# MAGIC 
# MAGIC Finally the best fit models will be selected with their tuned hyperparameters and then the test dataset is used to get the prediction score and other score metrics for the best model.

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

# MAGIC %md The overall goal of this report is to understand the data for the commodity indices and market information and define machine learning tools such as classes, functions, pipeline etc which will then be used in pre-procesing the data and designing statistical models to predict the target variable. A number of regression models will be investigated and then fine tune their hyperparameters to get the best hyperparameters by comparing the `mean_test_scores` for all the models which are cross-validated with the training dataset. 