# Databricks notebook source
# MAGIC %md #MA707 Report - Class Demonstrations (spring 2019, DataHeroes)

# COMMAND ----------

# MAGIC %md ## Introduction

# COMMAND ----------

# MAGIC %md In this notebook all the classes which will be used by the feature union and pipeline class during pre-processing of the dataset. They are fited with relevant variables from the cleaned dataset to get the transformed dataframe which will be fitted into the feature union.
# MAGIC 
# MAGIC The classes to be used during the pre-processing of the dataset are:
# MAGIC   - `CreateTargetDF` : Assigns target variable from the merged datasets
# MAGIC   - `CreateDatetimeVarsDF`: Creates new variables from the existing `datetime` variable by splitting into days, week, year, time etc.
# MAGIC   - `CreateLagVarsDF`: Creates lagged versions of all the feature variables to be used in the training model to predict using the un-lagged target variable 
# MAGIC   - `DropNaRowsDF`: Drops all rows if there is any missing values `NaN` in the dataset
# MAGIC   - `CountVectColDF`: Converts the content in a feature variable into individual tokens and counts the frequency of each tokens in each observation.
# MAGIC   - `TfidfVectColDF`: Converts the content in a feature variable into individual tokens and counts frequency in each observation and multiplied with the inverse occurance frequency throughout the whole rows observations in the variable.
# MAGIC   
# MAGIC ***Note These classes has been coded and explained in Notebook 0.2 Feature Creation***

# COMMAND ----------

# MAGIC %md ## Contents
# MAGIC 1. Setup
# MAGIC 2. Class Demonstrations
# MAGIC 3. Summary

# COMMAND ----------

# MAGIC %md ## 1. Setup

# COMMAND ----------

# MAGIC %run "./0.1 Raw dataset (inc)"

# COMMAND ----------

# MAGIC %run "./0.2 Feature creation (inc)"

# COMMAND ----------

# MAGIC %run "./0.3 Feature selection (inc)"

# COMMAND ----------

# MAGIC %run "./0.4 Estimators (inc)"

# COMMAND ----------

# MAGIC %run "./0.5 Pipeline functions (inc)"

# COMMAND ----------

# MAGIC %md ## 2. Class demonstrations

# COMMAND ----------

# MAGIC %md The following subsections demonstrate the classes used by the `FeatureUnion` and `Pipeline` classes to create a feature-target dataframe. 

# COMMAND ----------

# MAGIC %md ### 2.1 `CreateTargetDF`

# COMMAND ----------

# MAGIC %md This code fits the dataset `bci_dual_pdf` into the class `CreateTargetVarDF` defined in the notebook `./0.2 Feature creation (inc)` which takes the variable `bci_5tc` as its parameter and assigns it as the target variable.

# COMMAND ----------

# MAGIC %python
# MAGIC CreateTargetVarDF(var='bci_5tc') \
# MAGIC   .fit_transform(bci_dual_pdf) \
# MAGIC   .head()

# COMMAND ----------

# MAGIC %md Using the pipe operator, the dataset `bci_dual_pdf` is fitted into the defined class `CreateTargetVarDF` and then transformed to return the `bci_5tc` coulmn as the target variable using the `fit_transform` method. 

# COMMAND ----------

# MAGIC %md ### 2.2 `CreateDatetimeVarsDF`

# COMMAND ----------

# MAGIC %md Using the class `CreateDateTimeVarsDF` defined in the notebook `./0.2 Feature creation (inc)`, the variable `date` which is a `datetime` variable is fitted into the class and then transformed into new variables of the year, month, day, dayofyear week of the year and weekday using the `.dt` method.

# COMMAND ----------

bci_dual_pdf.info()

# COMMAND ----------

# MAGIC %python
# MAGIC CreateDatetimeVarsDF(var='date',
# MAGIC                      var_list=['year','month','day',
# MAGIC                                'dayofyear','weekofyear','weekday']) \
# MAGIC   .fit_transform(bci_dual_pdf) \
# MAGIC   .head()

# COMMAND ----------

# MAGIC %md The output is the newly created 6 variables from the `datetime` variable column `date`.

# COMMAND ----------

# MAGIC %md ### 2.3 `CreateLagVarsDF`

# COMMAND ----------

# MAGIC %md The below section creates a lagged version of all the variables other than `bci_5tc` which is the target variable in the data set `bci_coal_pdf`. 
# MAGIC The class `CreateLagVarsDF` takes two parameters `var_list` and `lag_list` which is the number of rows to be lagged and given the range `(0,2)` and then it creates a lagged version of the variables in the list `var_list`. It then returns the dataframe `bci_coal_pdf` with the lagged version of the variables concated into the existing dataframe. 

# COMMAND ----------

# MAGIC %python
# MAGIC CreateLagVarsDF(var_list=['cme_ln2','rici','p1a_03','p4_03','c7',
# MAGIC                           'cme_ln3','p3a_03','shfe_rb3','shfe_al3',
# MAGIC                           'shfe_cu3','ice_tib3','cme_fc3','opec_orb',
# MAGIC                           'ice_sb3','p3a_iv','ice_kc3','c5',
# MAGIC                           'p2a_03','cme_lc2','content','cme_sm3',
# MAGIC                           'ice_tib4','bci','tags','cme_ln1','cme_s2'],
# MAGIC                 lag_list=range(0,2)) \
# MAGIC   .fit_transform(bci_coal_pdf) \
# MAGIC   .loc[:5,['bci_lag0','bci_lag1']] \
# MAGIC   .head()

# COMMAND ----------

# MAGIC %md The output is the first 5 rows of the two lagged version of the variable `bci` with `bci_lag0` and `bci_lag1` which are lagged by zero and one respectively. 

# COMMAND ----------

bci_dual_pdf \
  .loc[:,['date','bci']] \
  .head()

# COMMAND ----------

# MAGIC %md ### 2.3 `DropNaRowsDF`

# COMMAND ----------

# MAGIC %md Create a pipeline `xfm_pipe` with only a single object `lag` which is the class `CreateLagVarsDF` which takes all the variables in the `var_list` and concats the 3 lagged versions of all the variables back into the dataframe. 
# MAGIC 
# MAGIC Fit the pipeline with the dataframe `bci_pdf` and return the transformed dataframe with the lagged variables concatenated to it. 

# COMMAND ----------

# MAGIC %python 
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC xfm_pipe = Pipeline(
# MAGIC   steps=[('lag',CreateLagVarsDF(var_list=['cme_ln2','rici','p1a_03','p4_03','c7','cme_ln3','p3a_03','shfe_rb3','shfe_al3',
# MAGIC                                           'shfe_cu3','ice_tib3','cme_fc3','opec_orb','ice_sb3','p3a_iv','ice_kc3','c5',
# MAGIC                                           'p2a_03','cme_lc2','content','cme_sm3','ice_tib4','bci','tags','cme_ln1','cme_s2'],
# MAGIC                                 lag_list=range(0,3)))
# MAGIC         ])
# MAGIC xfm_pipe \
# MAGIC   .fit_transform(bci_pdf) \
# MAGIC   .loc[:,['bci_lag0',
# MAGIC           'bci_lag1',
# MAGIC           'bci_lag2']] \
# MAGIC   .head(3)

# COMMAND ----------

# MAGIC %md The output shows the `bci` variable with its lagged versions with 0, 1 and 2 lagged and labelled as is. The entries in `bci` moves one row behind as it can be seen, the first value `3390.0` moves to the second index and third for the `bci_lag2` and it is replaced by `NaN`. These lagged predictor values will be used in training the data with the non lagged target variable.

# COMMAND ----------

# MAGIC %md Similar to above code, a new object `row` which is the class `DropNaRowsDF` is added to the existing pipeline `xfm_pipe` The class `DropNaRowsDF` deletes all the rows with any of its entry being `NaN` or missing which is assigned by `how='any'`. 
# MAGIC 
# MAGIC Fit the dataframe `bci_pdf` into the pipeline and print the first 2 rows of the lagged version of `bci`.

# COMMAND ----------

# MAGIC %python 
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC xfm_pipe = Pipeline(
# MAGIC   steps=[('lag',CreateLagVarsDF(var_list=['cme_ln2','rici','p1a_03','p4_03','c7','cme_ln3','p3a_03','shfe_rb3','shfe_al3',
# MAGIC                                           'shfe_cu3','ice_tib3','cme_fc3','opec_orb','ice_sb3','p3a_iv','ice_kc3','c5',
# MAGIC                                           'p2a_03','cme_lc2','content','cme_sm3','ice_tib4','bci','tags','cme_ln1','cme_s2'],
# MAGIC                                 lag_list=range(0,3))),
# MAGIC          ('row',DropNaRowsDF(how='any'))
# MAGIC         ])
# MAGIC xfm_pipe \
# MAGIC   .fit_transform(bci_pdf) \
# MAGIC   .loc[:,['bci_lag0',
# MAGIC           'bci_lag1',
# MAGIC           'bci_lag2']] \
# MAGIC   .head(2)

# COMMAND ----------

# MAGIC %md As `xfm_pipe` is a pipeline, when we fit it, it first fits the class `CreateLagVarsDF` which then transforms and returns a dataframe with all the lagged variables concatenated to the existing. It then fits this transformed lagged dataframe into the second class `DropNaRowsDF` which then drops the rows with any missing or `NaN` values in the columns.
# MAGIC 
# MAGIC The output is the three lagged version of the variable `bci`. 

# COMMAND ----------

# MAGIC %md ### 2.4 `CountVectColDF`

# COMMAND ----------

# MAGIC %md Define a class `CountVectColDF` which has a baseclass of `CountVectorizer`. It takes the parameters `col_name` which is the column name this class needs to fit into. It also takes a list of `ENGLISH_STOP_WORDS` as its parameter `stop_words` and new stop word list to be included as its parameter. The `super()` is used to call `__init__` method of the baseclass `CountVectorizer` which converts each words into a token and counts the number of tokens in each document or rows in that column.
# MAGIC 
# MAGIC The `fit` function fits the column name into the `super()` and then it transforms this fitted column and returns a dataframe using the `pd.Dataframe` which has new columns as feature names created from the fitted `CountVectorizer` class.

# COMMAND ----------

# MAGIC %python 
# MAGIC from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
# MAGIC class CountVectColDF(CountVectorizer):
# MAGIC   def __init__(self,col_name,prefix='cnt_',
# MAGIC                stop_words=list(ENGLISH_STOP_WORDS),
# MAGIC                add_stop_words=[]
# MAGIC               ):
# MAGIC     stop_words_list = stop_words+add_stop_words
# MAGIC     self.col_name = col_name
# MAGIC     self.prefix   = prefix
# MAGIC     super().__init__(stop_words=stop_words_list)
# MAGIC     return
# MAGIC   
# MAGIC   def fit(self,X,y=None):
# MAGIC     super().fit(X[self.col_name])
# MAGIC     return self
# MAGIC   
# MAGIC   def transform(self,X,y=None):
# MAGIC     return pd.DataFrame(data=super().transform(X[self.col_name]).toarray(),
# MAGIC                         columns=[self.prefix+feature_name for feature_name in super().get_feature_names()]
# MAGIC                        )

# COMMAND ----------

# MAGIC %md In this code, use the `tag` column from the fitted dataframe `bci_dual_pdf` as the parameter `col_name` into the above defined class `CountVectColDF`. The word `2012` is added to the `stop_words_list` which will be added to the list of `ENGLISH_STOP_WORDS`. The dataframe `bci_coal_pdf` is fitted and transformed to create new features or variables with the prefix `cnt_` and it then prints out the column names in the dataframe.

# COMMAND ----------

# MAGIC %python
# MAGIC CountVectColDF(col_name='tags_coal',
# MAGIC                prefix='cnt_',
# MAGIC                add_stop_words=['2012']) \
# MAGIC   .fit(bci_dual_pdf) \
# MAGIC   .transform(bci_dual_pdf) \
# MAGIC   .head() \
# MAGIC   .columns

# COMMAND ----------

# MAGIC %md The output is the list of all the feature names or tokens created from the `CountVectorizer` baseclass with the prefix `cnt`

# COMMAND ----------

# MAGIC %md ### 2.5 `TfidfVectColDF` 

# COMMAND ----------

# MAGIC %md Similar to `CountVectColDF`, the `TfidfVectColDF` has the baseclass `TfidfVectorizer` which counts the number of tokens in each document multiplied by the weight representing how common a word is across documents or different texts in the column. 

# COMMAND ----------

# MAGIC %python 
# MAGIC from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
# MAGIC class TfidfVectColDF(TfidfVectorizer):
# MAGIC   def __init__(self,col_name,prefix='tfidf_',
# MAGIC                stop_words=list(ENGLISH_STOP_WORDS),
# MAGIC                add_stop_words=[]
# MAGIC               ):
# MAGIC     stop_words_list = stop_words+add_stop_words
# MAGIC     self.col_name = col_name
# MAGIC     self.prefix   = prefix
# MAGIC     super().__init__(stop_words=stop_words_list)
# MAGIC     return
# MAGIC   
# MAGIC   def fit(self,X,y=None):
# MAGIC     super().fit(X[self.col_name])
# MAGIC     return self
# MAGIC   
# MAGIC   def transform(self,X,y=None):
# MAGIC     return pd.DataFrame(data=super().transform(X[self.col_name]).toarray(),
# MAGIC                         columns=[self.prefix+feature_name for feature_name in super().get_feature_names()]
# MAGIC                        )

# COMMAND ----------

# MAGIC %md Fit the dataframe `bci_dual_pdf` into the class `TfidfVectColDF` with the column name `tags` as the column to be fitted and transformed. It then returns the dataframe with all the feature names created by the `TfidfVectorizer` baseclass with the `Tf-idf` values and prints the first 5 rows of the new dataframe.

# COMMAND ----------

# MAGIC %python
# MAGIC TfidfVectColDF(col_name='tags_coal',
# MAGIC                prefix='tfidf_',
# MAGIC                add_stop_words=['2012']) \
# MAGIC   .fit(bci_dual_pdf) \
# MAGIC   .transform(bci_dual_pdf) \
# MAGIC   .head() 

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

# MAGIC %md All the above defined class will be used in creating feature union and pipeline creation while working with the three mining datasets and performing the gridsearch on various models to predict the target variable.