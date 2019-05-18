# Databricks notebook source
# MAGIC %md #MA707 Report - Preprocessing (spring 2019, DataHeroes)

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC This notebook contains code to create feature-target datasets. Here the two merged dataframes `bci_coal_pdf` and `bci_ironore_pdf` will go through pre-processing using the `FeatureUnionDF` class which conatins several `Feature Selection` and `Feature Creation` classes to provide a clean and complete dataset which will be used to train and test and investigate the regression models later in the report. Inorder to provide a better prediction of the response variable, all the feature variables are lagged with certain numbers.
# MAGIC 
# MAGIC These datasets will be split into training and test datasets which will then be used in cross-validation of the models.

# COMMAND ----------

# MAGIC %md ## Contents
# MAGIC 1. Setup
# MAGIC 2. Pipeline creation
# MAGIC 2. Create train and test datasets
# MAGIC 3. Summary

# COMMAND ----------

# MAGIC %md ## 1. Setup

# COMMAND ----------

# MAGIC %run "./1. Class demonstrations"

# COMMAND ----------

def display_pdf(a_pdf):
  display(spark.createDataFrame(a_pdf))

# COMMAND ----------

def est_grid_results_pdf(my_est_grid_obj,est_tag=None,fea_tag=None): 
  import pandas as pd
  import numpy  as np
  res_pdf = pd.DataFrame(data=my_est_grid_obj.cv_results_) \
           .loc[:,lambda df: np.logical_or(df.columns.str.startswith('param_'),
                                           df.columns.str.endswith('test_score'))
               ] \
           .loc[:,lambda df: np.logical_not(df.columns.str.startswith('split'))
               ] \
           .drop(['std_test_score'], 
                 axis=1)
  res_pdf.columns = [column.replace('param_','') for column in list(res_pdf.columns)]
  if est_tag is not None: res_pdf = res_pdf.assign(est_tag=est_tag)
  if fea_tag is not None: res_pdf = res_pdf.assign(fea_tag=fea_tag)
  return res_pdf.sort_values('mean_test_score')

# COMMAND ----------

# MAGIC %md In the above code `est_grid_results_pdf()` function is created which takes the sklearn GridSearchCV output as input and performs the below operations:
# MAGIC 
# MAGIC - Using the attribute `cv_results` of the GridSearchCV function which returns all the test scores as a numpy dictionary for any given execution and save the results in a pandas dataframe using `pd.DataFrame()` function.
# MAGIC - Using the resulting dataframe from previous step it checks all the columns and selects only columns which ends with `_test_score` or columns starting with `param_` to a new dataframe. It then drops the column `std_test_score` from the dataframe and finally returns the gridsearch output as a pandas dataframe using function with all the test score and ranks based on all assigned hyperparameters.

# COMMAND ----------

# MAGIC %md ### 2. Pipeline Creation

# COMMAND ----------

# MAGIC %md Create a function `get_count_vect_ore_three_plus_all_ts_pipe()` which returns a pipeline which then will be used to return a processed dataframe after performing all the pre-processing transformations for the merged dataframe `bci_ironore_pdf`. 

# COMMAND ----------

# MAGIC %python 
# MAGIC def get_count_vect_ore_three_plus_all_ts_pipe():
# MAGIC   from sklearn.pipeline import FeatureUnion, Pipeline
# MAGIC   return Pipeline(steps=[
# MAGIC     ('fea_one', FeatureUnionDF(transformer_list=[
# MAGIC       ('tgt_var'    ,CreateTargetVarDF(var='bci_5tc')),
# MAGIC       ('dt_vars'    ,CreateDatetimeVarsDF(var='date')),
# MAGIC       ('lag_ts_vars',CreateLagVarsDF(
# MAGIC         var_list=['cme_ln2','rici','p1a_03','p4_03','c7',
# MAGIC                   'cme_ln3','p3a_03','shfe_rb3','shfe_al3',
# MAGIC                   'shfe_cu3','ice_tib3','cme_fc3','opec_orb',
# MAGIC                   'ice_sb3','p3a_iv','ice_kc3','c5',
# MAGIC                   'p2a_03','cme_lc2','cme_sm3','ice_tib4','bci','cme_ln1','cme_s2'],
# MAGIC         lag_list=[3])),
# MAGIC       ('lag_txt_vars',CreateLagVarsDF(var_list=['tags_ore','content_ore','title_ore'],
# MAGIC                                       lag_list=[3])),
# MAGIC     ])),
# MAGIC     ('drop_na_rows'  ,DropNaRowsDF(how='any')),
# MAGIC     ('fea_two', FeatureUnionDF(transformer_list=[
# MAGIC       ('named_vars' ,CreateNamedVarsDF(except_list=['tags_ore_lag3','content_ore_lag3','title_ore_lag3'])),
# MAGIC       ('cnt_tags'   , CountVectColDF(col_name=   'tags_ore_lag3',prefix='cnt_ore_tags_'   ,add_stop_words=[])),
# MAGIC       ('cnt_content', CountVectColDF(col_name='content_ore_lag3',prefix='cnt_ore_content_',add_stop_words=[])),  
# MAGIC       ('cnt_title'  , CountVectColDF(col_name=  'title_ore_lag3',prefix='cnt_ore_title_'  ,add_stop_words=[])),  
# MAGIC     ])),
# MAGIC     ('drop_na_rows_again'  ,DropNaRowsDF(how='any')),
# MAGIC   ])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC `FeatureUnion` combines several transformer objects into a new transformer that combines their output. 
# MAGIC 
# MAGIC Above function `get_count_vect_ore_three_plus_all_ts_pipe()` code returns a pipeline which have two feature unions `fea_one` and `fea_two`. 
# MAGIC 
# MAGIC The first feature union `fea_one` dose the below operations.  
# MAGIC  - The object `tgt_var` which uses the `CreateTargetVarDF()` method to create target dataset with column `bci_5tc` using the input dataframe. 
# MAGIC  - Then object `dt_vars` which uses the `CreateDatetimeVarsDF()` method it takes `date` columns as input and creates many individual columns of `year`, `month`, `day`,          `hour`, `weekofyear`,`weekday` and `dayofyear`.    
# MAGIC  - Next object `lag_ts_vars` which uses the `CreateLagVarsDF()` method using the input columns names from the `var_list` and create the new time series lag variable list of      all of them with the number of days given in the `lag_list`. 
# MAGIC  - Next object `lag_txt_vars` which uses the `CreateLagVarsDF()` method using the input columns names from the `var_list` and create the new text lag variable list of all of    them with the number of days given in the `lag_list`. 
# MAGIC 
# MAGIC The Output of the first feature union is passed to object `drop_na_rows` which uses the `DropNaRowsDF()` method which drops all the rows which have any missing data. 
# MAGIC 
# MAGIC The second feature union `fea_two` takes the cleaned dataframe with no missing values created from the output of the method `DropNaRowsDF()` and does the below operations. 
# MAGIC  - The object `named_vars` uses the `CreateNamedVarsDF()` method using the columns names from the list `execpt_list` and excludes these columns from the original dataframe column list. 
# MAGIC  - Then object `cnt_tags` which uses the `CountVectColDF()` method creates new features also known as tokens for each individual document and prints the counts of each tokens in every row in the defined variable column into an array with the columns names prefixing with `cnt_tags_`. It also removes the tokens/variables for the words mentioned in the list `add_stop_words`. As a default list it takes the `sklearn stop word` list and it doesn't create new variables for them. The feature union then concatenates these newly created features into the dataframe.
# MAGIC  - The above steps are performed for all the three columns `tags_ore_lag3`, `content_ore_lag3` and `title_ore_lag3` and similarly all the newly created tokens/variables are concatenated to the existing dataframe which then returns the existing features with all the tokens and the count for each observations. 
# MAGIC  
# MAGIC The Output of the second feature union is passed to object `drop_na_rows` which uses the `DropNaRowsDF()` method which drops all rows with any missing data. 

# COMMAND ----------

# MAGIC %md Create another function `get_tfidf_vect_ore_three_plus_all_ts_pipe()` to perform the same operation on the merged dataframe `bci_ironore_pdf` but using the `TfidfVectColDF` instead of the `CountVectColDF` which produces values of the product of the **term-frequency and inverse document frequency** for the newly created tokenized variables and concatenated to the dataframe with the lagged variables.

# COMMAND ----------

# MAGIC %python 
# MAGIC def get_tfidf_vect_ore_three_plus_all_ts_pipe():
# MAGIC   from sklearn.pipeline import FeatureUnion, Pipeline
# MAGIC   return Pipeline(steps=[
# MAGIC     ('fea_one', FeatureUnionDF(transformer_list=[
# MAGIC       ('tgt_var'    ,CreateTargetVarDF(var='bci_5tc')),
# MAGIC       ('dt_vars'    ,CreateDatetimeVarsDF(var='date')),
# MAGIC       ('lag_ts_vars',CreateLagVarsDF(
# MAGIC         var_list=['cme_ln2','rici','p1a_03','p4_03','c7',
# MAGIC                   'cme_ln3','p3a_03','shfe_rb3','shfe_al3',
# MAGIC                   'shfe_cu3','ice_tib3','cme_fc3','opec_orb',
# MAGIC                   'ice_sb3','p3a_iv','ice_kc3','c5',
# MAGIC                   'p2a_03','cme_lc2','cme_sm3','ice_tib4','bci','cme_ln1','cme_s2'],
# MAGIC         lag_list=[3])),
# MAGIC       ('lag_txt_vars',CreateLagVarsDF(var_list=['tags_ore','content_ore','title_ore'],
# MAGIC                                       lag_list=[3])),
# MAGIC     ])),
# MAGIC     ('drop_na_rows'  ,DropNaRowsDF(how='any')),
# MAGIC     ('fea_two', FeatureUnionDF(transformer_list=[
# MAGIC       ('named_vars' ,CreateNamedVarsDF(except_list=['tags_ore_lag3','content_ore_lag3','title_ore_lag3'])),
# MAGIC       ('tfidf_tags'   , TfidfVectColDF(col_name=   'tags_ore_lag3',prefix='tfidf_tags_'   ,add_stop_words=[])),
# MAGIC       ('tfidf_content', TfidfVectColDF(col_name='content_ore_lag3',prefix='tfidf_content_',add_stop_words=[])),  
# MAGIC       ('tfidf_title'  , TfidfVectColDF(col_name=  'title_ore_lag3',prefix='tfidf_title_'  ,add_stop_words=[])),    
# MAGIC     ])),
# MAGIC     ('drop_na_rows_again'  ,DropNaRowsDF(how='any')),
# MAGIC   ])

# COMMAND ----------

# MAGIC %md The function `get_tfidf_vect_coal_three_plus_all_ts_pipe()` below returns a pipeline to pre-process the merged dataframe `bci_coal_pdf` and return the pre-processed dataframe with all the existing features as well as the newly created variables created from the tokenization of the `tag_coal`, `content_coal` and `title_coal` columns with their respective `tfidf values`

# COMMAND ----------

# MAGIC %python 
# MAGIC def get_tfidf_vect_coal_three_plus_all_ts_pipe():
# MAGIC   from sklearn.pipeline import FeatureUnion, Pipeline
# MAGIC   return Pipeline(steps=[
# MAGIC     ('fea_one', FeatureUnionDF(transformer_list=[
# MAGIC       ('tgt_var'    ,CreateTargetVarDF(var='bci_5tc')),
# MAGIC       ('dt_vars'    ,CreateDatetimeVarsDF(var='date')),
# MAGIC       ('lag_ts_vars',CreateLagVarsDF(
# MAGIC         var_list=['cme_ln2','rici','p1a_03','p4_03','c7',
# MAGIC                   'cme_ln3','p3a_03','shfe_rb3','shfe_al3',
# MAGIC                   'shfe_cu3','ice_tib3','cme_fc3','opec_orb',
# MAGIC                   'ice_sb3','p3a_iv','ice_kc3','c5',
# MAGIC                   'p2a_03','cme_lc2','cme_sm3','ice_tib4','bci','cme_ln1','cme_s2'],
# MAGIC         lag_list=[3])),
# MAGIC       ('lag_txt_vars',CreateLagVarsDF(var_list=['tags_coal','content_coal','title_coal'],
# MAGIC                                       lag_list=[3])),
# MAGIC     ])),
# MAGIC     ('drop_na_rows'  ,DropNaRowsDF(how='any')),
# MAGIC     ('fea_two', FeatureUnionDF(transformer_list=[
# MAGIC       ('named_vars' ,CreateNamedVarsDF(except_list=['tags_coal_lag3','content_coal_lag3','title_coal_lag3'])),
# MAGIC       ('tfidf_tags'   , TfidfVectColDF(col_name=   'tags_coal_lag3',prefix='tfidf_tags_'   ,add_stop_words=[])),
# MAGIC       ('tfidf_content', TfidfVectColDF(col_name='content_coal_lag3',prefix='tfidf_content_',add_stop_words=[])),  
# MAGIC       ('tfidf_title'  , TfidfVectColDF(col_name=  'title_coal_lag3',prefix='tfidf_title_'  ,add_stop_words=[])),    
# MAGIC     ])),
# MAGIC     ('drop_na_rows_again'  ,DropNaRowsDF(how='any')),
# MAGIC   ])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The above function `get_tfidf_vect_coal_three_plus_all_ts_pipe()` also returns a pipeline which have two feature unions `fea_one` and `fea_two` which then will return a dataframe with all the lagged version of the variables in the `bci_pdf` and the new features created by tokenizing the `tag`, `content`, `title` columns concatenated to it with all `tfidf` values as their values. It also contains features created from the `datetime` variable as the `year`, `month`, `day`, `hour`, `weekofyear`,`weekday` and `dayofyear`.
# MAGIC 
# MAGIC It will return the cleaned pre-processed dataset by removing all missing values if we fit the merged dataframe `bci_coal_pdf`.

# COMMAND ----------

# MAGIC %md Similarly the below function creates a pipeline to transform the `bci_coal_pdf` dataframe with lagged variables alongwith the new tokeneized variables and their frequency count. It then removes all rows with missing values using the `drop_na_rows_again` object which drops all rows with any missing value set by the parameter in the method `DropNaRowsDF(how='any')`.

# COMMAND ----------

# MAGIC %python 
# MAGIC def get_count_vect_coal_three_plus_all_ts_pipe():
# MAGIC   from sklearn.pipeline import FeatureUnion, Pipeline
# MAGIC   return Pipeline(steps=[
# MAGIC     ('fea_one', FeatureUnionDF(transformer_list=[
# MAGIC       ('tgt_var'    ,CreateTargetVarDF(var='bci_5tc')),
# MAGIC       ('dt_vars'    ,CreateDatetimeVarsDF(var='date')),
# MAGIC       ('lag_ts_vars',CreateLagVarsDF(
# MAGIC         var_list=['cme_ln2','rici','p1a_03','p4_03','c7',
# MAGIC                   'cme_ln3','p3a_03','shfe_rb3','shfe_al3',
# MAGIC                   'shfe_cu3','ice_tib3','cme_fc3','opec_orb',
# MAGIC                   'ice_sb3','p3a_iv','ice_kc3','c5',
# MAGIC                   'p2a_03','cme_lc2','cme_sm3','ice_tib4','bci','cme_ln1','cme_s2'],
# MAGIC         lag_list=[3])),
# MAGIC       ('lag_txt_vars',CreateLagVarsDF(var_list=['tags_coal','content_coal','title_coal'],
# MAGIC                                       lag_list=[3])),
# MAGIC     ])),
# MAGIC     ('drop_na_rows'  ,DropNaRowsDF(how='any')),
# MAGIC     ('fea_two', FeatureUnionDF(transformer_list=[
# MAGIC       ('named_vars' ,CreateNamedVarsDF(except_list=['tags_coal_lag3','content_coal_lag3','title_coal_lag3'])),
# MAGIC       ('cnt_tags'   , CountVectColDF(col_name=   'tags_coal_lag3',prefix='cnt_coal_tags_'   ,add_stop_words=[])),
# MAGIC       ('cnt_content', CountVectColDF(col_name='content_coal_lag3',prefix='cnt_coal_content_',add_stop_words=[])),  
# MAGIC       ('cnt_title'  , CountVectColDF(col_name=  'title_coal_lag3',prefix='cnt_coal_title_'  ,add_stop_words=[])),  
# MAGIC     ])),
# MAGIC     ('drop_na_rows_again'  ,DropNaRowsDF(how='any')),
# MAGIC   ])

# COMMAND ----------

# MAGIC %md ### Fitting datasets into the pipeline functions

# COMMAND ----------

# MAGIC %md #### Dataframe 1: `fea_tgt_coal_tfidf_pdf`
# MAGIC Fit the merged dataframe `bci_coal_pdf` into the above defined function `get_tfidf_vect_coal_three_plus_all_ts_pipe()` and create a new dataframe `fea_tgt_coal_tfidf_pdf` which cotains the lagged time series feature values and the features created by `TfidfVectorizer` method with the `tf-idf` values of the extracted variables from the text columns in the dataframe, and all the NA rows dropped.

# COMMAND ----------

fea_tgt_coal_tfidf_pdf = \
  get_tfidf_vect_coal_three_plus_all_ts_pipe() \
  .fit(bci_coal_pdf) \
  .transform(bci_coal_pdf)

# COMMAND ----------

fea_tgt_coal_tfidf_pdf.info()

# COMMAND ----------

# MAGIC %md The new processed dataframe `fea_tgt_coal_tfidf_pdf` contains 1591 rows with 41,766 variables in it. This dataframe will be used later to train and test different regression models to get the best fitted model to predict the `bci_5tc` value.

# COMMAND ----------

# MAGIC %md #### Dataframe 2: `fea_tgt_coal_cnt_pdf`
# MAGIC Similarly, the `bci_coal_pdf` dataframe is fitted into the second function `get_count_vect_coal_three_plus_all_ts_pipe()` which then returns the dataframe `fea_tgt_coal_cnt_pdf` containing the lagged time series features as well the tokenized features with their count frequency in each document or observations from the text columns in the dataframe.

# COMMAND ----------

fea_tgt_coal_cnt_pdf = \
  get_count_vect_coal_three_plus_all_ts_pipe() \
  .fit(bci_coal_pdf) \
  .transform(bci_coal_pdf)

# COMMAND ----------

fea_tgt_coal_cnt_pdf.info()

# COMMAND ----------

# MAGIC %md #### Dataframe 3: `fea_tgt_ore_tfidf_pdf`
# MAGIC Perfom the same `fit` and `transform` operations on the `bci_iron_ore` dataframe and create two new dataframes with one having the count of tokens in each documents concatenated as features to the lagged time series features while the other the product `tf-idf` value for each tokens in every documents in the text columns of the dataframe.

# COMMAND ----------

fea_tgt_ore_tfidf_pdf = \
  get_tfidf_vect_ore_three_plus_all_ts_pipe() \
  .fit(bci_ironore_pdf) \
  .transform(bci_ironore_pdf)

# COMMAND ----------

fea_tgt_ore_tfidf_pdf.info()

# COMMAND ----------

# MAGIC %md #### Dataframe 4: `fea_tgt_ore_cnt_pdf`

# COMMAND ----------

fea_tgt_ore_cnt_pdf = \
  get_count_vect_ore_three_plus_all_ts_pipe() \
  .fit(bci_ironore_pdf) \
  .transform(bci_ironore_pdf)

# COMMAND ----------

fea_tgt_ore_cnt_pdf.info()

# COMMAND ----------

# MAGIC %md The two new dataframes `fea_tgt_ore_tfidf_pdf` and `fea_tgt_ore_cnt_pdf` contains 1592 rows of observations with 39617 variables. These two will be used to train and test all models and select the best regression model.

# COMMAND ----------

# MAGIC %md ## 3. Create train and test datasets

# COMMAND ----------

def create_train_test_ts(fea_pdf, tgt_ser, trn_prop=0.8):
  trn_len = int(trn_prop * len(fea_pdf))
  return (fea_pdf.iloc[:trn_len],
          fea_pdf.iloc[ trn_len:],
          tgt_ser.iloc[:trn_len],
          tgt_ser.iloc[ trn_len:]
         )

# COMMAND ----------

# MAGIC %md The above function divides feature dataset and target dataset into training set and test set. The training percentage is defined as `trn_prop=0.8` which means 80% of the dataset will be used for training the model and the rest for testing the model. 
# MAGIC The traning row length `trn_len`is the integer of the total row number multiply training percentage. Then the training set subtract `trn_len` rows 
# MAGIC of the original feature and target datasets using `iloc` function.

# COMMAND ----------

# MAGIC %md Call the `create_train_test_ts()` function, fit the dataframe `fea_tgt_coal_tdif_pdf` with the `target` variable dropped as feature dataframe, and extract the varibale `target` from `fea_tgt_coal_tdif_pdf` as target dataset. Use function to split training set and test set on the feature dataset and target dataset.
# MAGIC 
# MAGIC Print the shape of the training and testing datasets created.

# COMMAND ----------

(trn_coal_tfidf_fea_pdf, tst_coal_tfidf_fea_pdf, 
 trn_coal_tfidf_tgt_ser, tst_coal_tfidf_tgt_ser
) = \
create_train_test_ts(fea_pdf = fea_tgt_coal_tfidf_pdf.drop( 'target',axis=1),
                     tgt_ser = fea_tgt_coal_tfidf_pdf.loc[:,'target'],
                    )

trn_coal_tfidf_fea_pdf.shape, tst_coal_tfidf_fea_pdf.shape, trn_coal_tfidf_tgt_ser.shape, tst_coal_tfidf_tgt_ser.shape

# COMMAND ----------

# MAGIC %md From the 1591 rows with 41765 variables, the training dataset will have 1272 observations with 41765 columns and the rest 319 rows as the testing dataset. As the target variable is a single dimensional variables, it is equally split into 1272 and 319 rows as training and testing sets respectively. 

# COMMAND ----------

# MAGIC %md Similar to the above train-test split of the `fea_tgt_coal_tfidf_pdf`, all the other three dataframes created from the featureunion are split using the `create_train_test_ts` function with a training percentage of 80%.

# COMMAND ----------

(trn_coal_cnt_fea_pdf, tst_coal_cnt_fea_pdf, 
 trn_coal_cnt_tgt_ser, tst_coal_cnt_tgt_ser
) = \
create_train_test_ts(fea_pdf = fea_tgt_coal_cnt_pdf.drop( 'target',axis=1),
                     tgt_ser = fea_tgt_coal_cnt_pdf.loc[:,'target'],
                    )

trn_coal_cnt_fea_pdf.shape, tst_coal_cnt_fea_pdf.shape, trn_coal_cnt_tgt_ser.shape, tst_coal_cnt_tgt_ser.shape

# COMMAND ----------

# MAGIC %md Similarly for the dataframe `fea_tgt_ore_tfidf_pdf` which has 1592 rows and 39616 variables, the train test split created 1273 observations for training the model and 319 observations to test the model for both the feature and target. The shape of the training and test feature and target dataset are printed accordingly.

# COMMAND ----------

(trn_ore_tfidf_fea_pdf, tst_ore_tfidf_fea_pdf, 
 trn_ore_tfidf_tgt_ser, tst_ore_tfidf_tgt_ser
) = \
create_train_test_ts(fea_pdf = fea_tgt_ore_tfidf_pdf.drop( 'target',axis=1),
                     tgt_ser = fea_tgt_ore_tfidf_pdf.loc[:,'target'],
                    )

trn_ore_tfidf_fea_pdf.shape, tst_ore_tfidf_fea_pdf.shape, trn_ore_tfidf_tgt_ser.shape, tst_ore_tfidf_tgt_ser.shape

# COMMAND ----------

(trn_ore_cnt_fea_pdf, tst_ore_cnt_fea_pdf, 
 trn_ore_cnt_tgt_ser, tst_ore_cnt_tgt_ser
) = \
create_train_test_ts(fea_pdf = fea_tgt_ore_cnt_pdf.drop( 'target',axis=1),
                     tgt_ser = fea_tgt_ore_cnt_pdf.loc[:,'target'],
                    )

trn_ore_cnt_fea_pdf.shape, tst_ore_cnt_fea_pdf.shape, trn_ore_cnt_tgt_ser.shape, tst_ore_cnt_tgt_ser.shape

# COMMAND ----------

# MAGIC %md ## 3. Summary

# COMMAND ----------

# MAGIC %md Several preprocessing pipelines have been used to create the feature dataset and target dataset. Then we split the pre-processed dataframe into the train and test datasets with 80% used to train the models and 20% to test them. 
# MAGIC 
# MAGIC Four sets of train test datasets formed by the different combination of the merged datasets: 
# MAGIC  - `bci_coal_pdf` with CountVectorizer 
# MAGIC  - `bci_coal_pdf` with TfidfVectorizer
# MAGIC  - `bci_ironore_pdf` with CountVectorizer 
# MAGIC  - `bci_ironore_pdf` with TfidfVectorizer
# MAGIC These combinations of train test dataset will be used in the estimator pipelines as well as in the gridsearch and compared with their scores.