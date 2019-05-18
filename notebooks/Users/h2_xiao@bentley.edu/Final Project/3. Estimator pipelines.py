# Databricks notebook source
# MAGIC %md #MA707 Report - Estimator Pipelines (spring 2019, DataHeroes)

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC This notebook contains code to create estimator pipelines which include the feature selection classes and the wrapped estimator classes. All the different train-test dataset defined in the `Preprocessing pipeline` will be fitted and tested to get their individual score at the default parameters as defined. 
# MAGIC The code will be reused in the `Investigation` notebook where it will be fitted to various pipelines with different hyperparameters in GridSeach and get the `mean_test_scores` and their rank. 
# MAGIC 
# MAGIC The train-test dataset used for fitting the models and predicting in this notebooks are:
# MAGIC - [Dataframe 1: `fea_tgt_coal_tfidf_pdf` ](https://bentley.cloud.databricks.com/#notebook/1364451/command/1603805)
# MAGIC - [Dataframe 2: `fea_tgt_coal_cnt_pdf`](https://bentley.cloud.databricks.com/#notebook/1364451/command/1603807)
# MAGIC - [Dataframe 3: `fea_tgt_ore_tfidf_pdf`](https://bentley.cloud.databricks.com/#notebook/1364451/command/1603808)
# MAGIC - [Dataframe 4: `fea_tgt_ore_cnt_pdf`](https://bentley.cloud.databricks.com/#notebook/1364451/command/1603809)

# COMMAND ----------

# MAGIC %md ## Contents
# MAGIC 1. Setup

# COMMAND ----------

# MAGIC %md ## 1. Setup

# COMMAND ----------

# MAGIC %run "./2. Preprocessing pipelines"

# COMMAND ----------

# MAGIC %md ### Lasso Pipelines

# COMMAND ----------

# MAGIC %md Least Absolute Shrinkage Selector Operator in short Lasso regression not only helps in reducing over-fitting but it can help in feature selection. It has the ability to shrink the estimated coefficient for the model to exactly zero, reducing the number of features and serve as a feature selection tool as well as the regression tool at the same time. By adding a biasing term to the regression optimization function to reduce the effect of collinearity. Lasso helps reduce the multicollinearity of endogenous variables in models which creates inaccurate estimates of the regression coefficients. 
# MAGIC 
# MAGIC The [EstimatorLasso()](https://bentley.cloud.databricks.com/#notebook/1607089/command/1607095) wrapper class have the default parameters `alpha=1`  and `normalize=True` so the regressors X will be normalized before regression. 
# MAGIC  
# MAGIC **Note:** In order to get the best value of `alpha`, the data in the training set will be used to get estimate coefficients’ value for every value of `alpha`. In the validation set, different values of the product of `alpha` and coefficient estimates will be assessed. The one which has lower error value will get selected. This selected value of coefficients’ will again be assessed by test data set.

# COMMAND ----------

# MAGIC %python 
# MAGIC def get_lasso_pipeline():
# MAGIC   from sklearn.pipeline import FeatureUnion, Pipeline
# MAGIC   from sklearn.linear_model import LinearRegression, Ridge
# MAGIC   from sklearn.feature_selection import chi2
# MAGIC   return Pipeline(steps=[
# MAGIC     ('lso', EstimatorLasso())
# MAGIC   ])

# COMMAND ----------

# MAGIC %md Fit the different train-test datasets into the above estimator pipeline and compare the scores.

# COMMAND ----------

get_lasso_pipeline() \
  .fit  (trn_coal_tfidf_fea_pdf, trn_coal_tfidf_tgt_ser) \
  .score(tst_coal_tfidf_fea_pdf, tst_coal_tfidf_tgt_ser)

# COMMAND ----------

get_lasso_pipeline() \
  .fit  (trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser) \
  .score(tst_coal_cnt_fea_pdf, tst_coal_cnt_tgt_ser)

# COMMAND ----------

get_lasso_pipeline() \
  .fit  (trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser) \
  .score(tst_ore_tfidf_fea_pdf, tst_ore_tfidf_tgt_ser)

# COMMAND ----------

get_lasso_pipeline() \
  .fit  (trn_ore_cnt_fea_pdf, trn_ore_cnt_tgt_ser) \
  .score(tst_ore_cnt_fea_pdf, tst_ore_cnt_tgt_ser)

# COMMAND ----------

# MAGIC %md The Lasso regression is performed on the 3 day lag data of `bci` and 3 day lag data of counts and frequency of `tags`, `title` and `content` of the different datasets created in the pre-processing noteboook. The table below shows the results for each dataset. <table>
# MAGIC     <tr>
# MAGIC         <td>`bci_coal_pdf`</td><td>`bci_ironore_pdf`</td><td>`CountVectorizer`</td><td>`TfidfVectorizer`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>Yes</td><td></td><td>Yes</td><td></td><td>72.21%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td></td><td>Yes</td><td>Yes</td><td></td><td>68.69%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>Yes</td><td></td><td></td><td>Yes</td><td>69.96</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td></td><td>Yes</td><td></td><td>Yes</td><td>71.80%</td>
# MAGIC     </tr>  
# MAGIC </table>
# MAGIC 
# MAGIC 
# MAGIC From the results, it can be seen that the train-test dataset created by performing the feature extraction `CountVectorizer` method on the `bci_coal_pdf` and `TfidfVectorizer` on the `bci_ironore_pdf` both have the R Square (R2) value close to 72%.  R-square is the coefficient of determination which determines the proportion of the variability explained by the model. Later in the investigation notebook, these two datasets will try to run `GridSearchCV` to fine tune the hyper parameters and improve the model efficiency. 

# COMMAND ----------

# MAGIC %md ##PCA Lasso Pipeline

# COMMAND ----------

# MAGIC %md Extending the Lasso model would be to add another feature reduction process to reduce the features and in process to check the model prediction.  
# MAGIC 
# MAGIC Add the [FeatureSelectionPCA](https://bentley.cloud.databricks.com/#notebook/1607074/command/1607080) wrapper class which has the base class of PCA (Principal component analysis). It is a technique for feature extraction — so it combines the input variables in a specific way, then can drop the “least important” variables while still retaining the most valuable parts of all of the variables. So here below creating a pipeline by using the `FeatureSelectionPCA()` wrapper class which have default parameters
# MAGIC  - `n_components=10` :- Means here it will create 10 new featuers using the existing features. 

# COMMAND ----------

# MAGIC %python 
# MAGIC def get_pca_lasso_pipeline():
# MAGIC   from sklearn.pipeline import FeatureUnion, Pipeline
# MAGIC   from sklearn.linear_model import LinearRegression, Ridge
# MAGIC   return Pipeline(steps=[
# MAGIC     ('pca', FeatureSelectionPCA()),
# MAGIC     ('lso', EstimatorLasso())
# MAGIC   ])

# COMMAND ----------

# MAGIC %md Fit the estimator pipeline with the two classes and compare the scores with the different training and testing sets.

# COMMAND ----------

get_pca_lasso_pipeline() \
  .fit  (trn_coal_tfidf_fea_pdf, trn_coal_tfidf_tgt_ser) \
  .score(tst_coal_tfidf_fea_pdf, tst_coal_tfidf_tgt_ser)

# COMMAND ----------

get_pca_lasso_pipeline() \
  .fit  (trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser) \
  .score(tst_coal_cnt_fea_pdf, tst_coal_cnt_tgt_ser)

# COMMAND ----------

get_pca_lasso_pipeline() \
  .fit  (trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser) \
  .score(tst_ore_tfidf_fea_pdf, tst_ore_tfidf_tgt_ser)

# COMMAND ----------

get_pca_lasso_pipeline() \
  .fit  (trn_ore_cnt_fea_pdf, trn_ore_cnt_tgt_ser) \
  .score(tst_ore_cnt_fea_pdf, tst_ore_cnt_tgt_ser)

# COMMAND ----------

# MAGIC %md %md PCA-Lasso pipeline performed on the below combination of datasets, the table below shows the results.  
# MAGIC  <table>
# MAGIC     <tr>
# MAGIC         <td>`bci_coal_pdf`</td><td>`bci_ironore_pdf`</td><td>`CountVectorizer`</td><td>`TfidfVectorizer`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>Yes</td><td></td><td>Yes</td><td></td><td>55.31%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td></td><td>Yes</td><td>Yes</td><td></td><td>55.21%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>Yes</td><td></td><td></td><td>Yes</td><td>57.05%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td></td><td>Yes</td><td></td><td>Yes</td><td>57.21%</td>
# MAGIC     </tr>  
# MAGIC </table>
# MAGIC 
# MAGIC As we see `bci_coal_pdf with CountVectorizer` and `bci_ironore_pdf with TfidfVectorizer` both have the R Square (R2) value of around 55% to 57%. Eventhough there is a drop in the variability with the addition of PCA, but as default only 10 features have been selected as default. This estimator pipeline will be tried  using `GridSearchCV` with addtion of many features by fine tuning the hyper parameters to check any improvement in the model efficiency. 

# COMMAND ----------

# MAGIC %md ### ElasticNet Pipelines

# COMMAND ----------

# MAGIC %md ElasticNet is a third commonly used model of regression which incorporates penalties from both Ridge and Lasso models. Ridge regression model is explained in the following sections. Addition to setting and choosing a lambda value elastic net also allows to tune the alpha parameter where 𝞪 = 0 corresponds to ridge and 𝞪 = 1 to lasso. Therefore an alpha value can be chosen between 0 and 1 to optimize the elastic net. ElasticNet keeps the group effect in the case of highly correlated variables, rather than zeroing some of them like Lasso and also no limitation on the number of selected variables. 
# MAGIC 
# MAGIC The [EstimatorElasticNet()](https://bentley.cloud.databricks.com/#notebook/1607089/command/1607099) wrapper class which have default parameters `alpha=0.5`it gives the equal weight to both Ridge and Lasso lambda values and `normalize=True` which regressors X will be normalized before regression. 
# MAGIC  
# MAGIC  **Note:** In order to get the best value of `alpha`, the data in the training set will be used to get estimate coefficients’ value for every value of `alpha`. In the validation set, different values of the product of `alpha` and coefficient estimates will be assessed. The one which has lower error value will get selected. This selected value of coefficients’ will again be assessed by test data set.

# COMMAND ----------

# MAGIC %md ElasticNet regression is performed on the below combination of datasets. 

# COMMAND ----------

# MAGIC %python 
# MAGIC def get_elasticnet_pipeline():
# MAGIC   from sklearn.pipeline import FeatureUnion, Pipeline
# MAGIC   from sklearn.linear_model import LinearRegression, Ridge
# MAGIC   return Pipeline(steps=[
# MAGIC     ('ela', EstimatorElasticNet())
# MAGIC   ])

# COMMAND ----------

get_elasticnet_pipeline() \
  .fit  (trn_coal_tfidf_fea_pdf, trn_coal_tfidf_tgt_ser) \
  .score(tst_coal_tfidf_fea_pdf, tst_coal_tfidf_tgt_ser)

# COMMAND ----------

get_elasticnet_pipeline() \
  .fit  (trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser) \
  .score(tst_coal_cnt_fea_pdf, tst_coal_cnt_tgt_ser)

# COMMAND ----------

get_elasticnet_pipeline() \
  .fit  (trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser) \
  .score(tst_ore_tfidf_fea_pdf, tst_ore_tfidf_tgt_ser)

# COMMAND ----------

get_elasticnet_pipeline() \
  .fit  (trn_ore_cnt_fea_pdf, trn_ore_cnt_tgt_ser) \
  .score(tst_ore_cnt_fea_pdf, tst_ore_cnt_tgt_ser)

# COMMAND ----------

# MAGIC  %md %md The table below shows the results (r-square value) for the different datasets fitted into the Elaastic net estimator pipeline.
# MAGIC   <table>
# MAGIC     <tr>
# MAGIC         <td>`bci_coal_pdf`</td><td>`bci_ironore_pdf`</td><td>`CountVectorizer`</td><td>`TfidfVectorizer`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>Yes</td><td></td><td>Yes</td><td></td><td> -1.595</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td></td><td>Yes</td><td>Yes</td><td></td><td>-1.566</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>Yes</td><td></td><td></td><td>Yes</td><td>-1.603</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td></td><td>Yes</td><td></td><td>Yes</td><td>-1.565</td>
# MAGIC     </tr>  
# MAGIC </table>
# MAGIC 
# MAGIC As we see for all the combinations have the R Square (R2) the proportion of the variability explained by the models is negative which means the model is not good. Will try to run `GridSearchCV` to check if any scope of improvement, also like to check the same by adding PCA too. Since all of the test scores are negative, we try Elastic Net model with PCA to improve the model performance
# MAGIC 
# MAGIC Extending the Elastic model by adding another feature reduction process PCA to check if the overall predicton will improve. 

# COMMAND ----------

# MAGIC %python 
# MAGIC def get_pca_elasticnet_pipeline():
# MAGIC   from sklearn.pipeline import FeatureUnion, Pipeline
# MAGIC   from sklearn.linear_model import LinearRegression, Ridge
# MAGIC   return Pipeline(steps=[
# MAGIC     ('pca', FeatureSelectionPCA()),
# MAGIC     ('ela', EstimatorElasticNet())
# MAGIC   ])

# COMMAND ----------

get_pca_elasticnet_pipeline() \
  .fit  (trn_coal_tfidf_fea_pdf, trn_coal_tfidf_tgt_ser) \
  .score(tst_coal_tfidf_fea_pdf, tst_coal_tfidf_tgt_ser)

# COMMAND ----------

get_pca_elasticnet_pipeline() \
  .fit  (trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser) \
  .score(tst_coal_cnt_fea_pdf, tst_coal_cnt_tgt_ser)

# COMMAND ----------

get_pca_elasticnet_pipeline() \
  .fit  (trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser) \
  .score(tst_ore_tfidf_fea_pdf, tst_ore_tfidf_tgt_ser)

# COMMAND ----------

get_pca_elasticnet_pipeline() \
  .fit  (trn_ore_cnt_fea_pdf, trn_ore_cnt_tgt_ser) \
  .score(tst_ore_cnt_fea_pdf, tst_ore_cnt_tgt_ser)

# COMMAND ----------

# MAGIC %md ElasticNet regression with PCA performed on the above combination of datasets. As noticed ealier the  R Square of the models is still negative which means the model is not good. 

# COMMAND ----------

# MAGIC %md ### SVR Pipelines

# COMMAND ----------

# MAGIC %md Support Vector Regression (SVR) uses the same principles as the SVM for classification, with only a few minor differences. In simple regression we try to minimise the error rate. While in SVR we try to fit the error within a certain threshold.  Objective with SVR is to basically consider the points that are within the boundary line, so best fit line is the line hyperplane that has maximum number of points.

# COMMAND ----------

# MAGIC %md Build the `get_svr_pipeline` function which returns a pipeline [Estimatorsvr](https://bentley.cloud.databricks.com/#notebook/1607089/command/1607103) which calls the base class `SVR` with default parameters `kernel='rbf', gamma='auto', C=100`. Call the function and fit the pipeline with
# MAGIC 4 different datasets then get the R score on the relative test datasets. Compare the scores.

# COMMAND ----------

# MAGIC %python 
# MAGIC def get_svr_pipeline():
# MAGIC   from sklearn.pipeline import FeatureUnion, Pipeline
# MAGIC   from sklearn.svm               import SVR
# MAGIC   from sklearn.feature_selection import chi2
# MAGIC   return Pipeline(steps=[
# MAGIC     ('svr', EstimatorSVR())
# MAGIC   ])

# COMMAND ----------

get_svr_pipeline() \
  .fit  (trn_coal_tfidf_fea_pdf, trn_coal_tfidf_tgt_ser) \
  .score(tst_coal_tfidf_fea_pdf, tst_coal_tfidf_tgt_ser)

# COMMAND ----------

get_svr_pipeline() \
  .fit  (trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser) \
  .score(tst_coal_cnt_fea_pdf, tst_coal_cnt_tgt_ser)

# COMMAND ----------

get_svr_pipeline() \
  .fit  (trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser) \
  .score(tst_ore_tfidf_fea_pdf, tst_ore_tfidf_tgt_ser)

# COMMAND ----------

get_svr_pipeline() \
  .fit  (trn_ore_cnt_fea_pdf, trn_ore_cnt_tgt_ser) \
  .score(tst_ore_cnt_fea_pdf, tst_ore_cnt_tgt_ser)

# COMMAND ----------

# MAGIC %md The scores are as following: -2.896, -2.896, -2.881, -2.881. The scores are all 
# MAGIC negative for the default parameters in the model, the model appears to be not working very well with the datasets. SVR regression also results in a R Square (R2) value which is negative so the model is not good to be used. So inorder to get better results from this model, it will be tested using the GridSearchCV with different parameters other than the default values during investigation. 

# COMMAND ----------

# MAGIC %md ###Ridge Pipelines

# COMMAND ----------

# MAGIC %md Ridge regression is a regression technique for continuous value prediction. It is a simple techniques to reduce model complexity and prevent over-fitting which may result from simple linear regression. Using a ridge regression method helps reduce the multicollinearity of endogenous variables in models. Multicollinearity creates inaccurate estimates of the regression coefficients.
# MAGIC 
# MAGIC Generally when overfitting happens, the regression coefficients’ values becomes very huge. Ridge regression is used to quantify the overfitting of the data through measuring the magnitude of coefficients.  The parameter `alpha` as defined in the wrapper class `EstimatorRidge` in *Notebook 0.4* is the tuning parameter to balance the fit of data and magnitude of coefficients.
# MAGIC 
# MAGIC The class [EstimatorRidge()](https://bentley.cloud.databricks.com/#notebook/1607089/command/1607091) which calls on the base class `Ridge` from sklearn.linear_model package, with all its default parameter `normalize=False` and `solver='auto'` which chooses the solver automatically based on the type of data.
# MAGIC 
# MAGIC **Note:** Inorder to get the best value of `alpha`, the data in the training set will be used to get estimate coefficients’ value for every value of `alpha`. In the validation set, different values of the product of `alpha` and coefficient estimates will be assessed. The one which has lower error value will get selected. This selected value of coefficients’ will again be assessed by test data set.

# COMMAND ----------

# MAGIC %md Build `get_ridge_pipeline` function, it returns a pipeline with the estimator class [estimatorRidge](https://bentley.cloud.databricks.com/#notebook/1607089/command/1607091). Call the function and fit the pipeline with
# MAGIC 4 different datasets then get the R score on the relative test datasets. 

# COMMAND ----------

def get_ridge_pipeline():
  from sklearn.pipeline import FeatureUnion, Pipeline
  from sklearn.linear_model import LinearRegression, Ridge
  return Pipeline(steps=[
    ('rdg', EstimatorRidge())
  ])

# COMMAND ----------

get_ridge_pipeline() \
  .fit  (trn_coal_tfidf_fea_pdf, trn_coal_tfidf_tgt_ser) \
  .score(tst_coal_tfidf_fea_pdf, tst_coal_tfidf_tgt_ser)

# COMMAND ----------

get_ridge_pipeline() \
  .fit  (trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser) \
  .score(tst_coal_cnt_fea_pdf, tst_coal_cnt_tgt_ser)

# COMMAND ----------

get_ridge_pipeline() \
  .fit  (trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser) \
  .score(tst_ore_tfidf_fea_pdf, tst_ore_tfidf_tgt_ser)

# COMMAND ----------

get_ridge_pipeline() \
  .fit  (trn_ore_cnt_fea_pdf, trn_ore_cnt_tgt_ser) \
  .score(tst_ore_cnt_fea_pdf, tst_ore_cnt_tgt_ser)

# COMMAND ----------

# MAGIC %md The scores are as following: -0.971, -0.835, -0.744, -0.549. The scores are all 
# MAGIC negative with the default parameters for the wrapper class. Hence in the Investigation notebook different hyperparameters will be tested to fine tune the model during gridsearch and results will be compared and ranked based on their `mean_test_score` value.

# COMMAND ----------

# MAGIC %md ### Decision Tree

# COMMAND ----------

# MAGIC %md A decision tree is a largely used non-parametric effective machine learning modeling technique for regression and classification problems. To find solutions a decision tree makes sequential, hierarchical decision about the outcomes variable based on the predictor data.
# MAGIC 
# MAGIC Hierarchical means the model is defined by a series of questions that lead to a class label or a value when applied to any observation. Once set up the model acts like a protocol in a series of “if this occurs then this occurs” conditions that produce a specific result from the input data.
# MAGIC 
# MAGIC A Non-parametric method means that there are no underlying assumptions about the distribution of the errors or the data. It basically means that the model is constructed based on the observed data.

# COMMAND ----------

# MAGIC %md Build the `get_DecisionTree_pipeline` function which returns a pipeline with an estimator [EstimatorDecisiontree](https://bentley.cloud.databricks.com/#notebook/1607089/command/1607111) which has a base class of `DecisionTreeRegressor` from `sklearn.tree`. The default parameter for the estimator is set as `max_depth=5, max_leaf_nodes=10`. Call the function and fit the pipeline with
# MAGIC 4 different datasets then get the R score on the relative test datasets. 

# COMMAND ----------

def get_DecisionTree_pipeline():
  from sklearn.pipeline import FeatureUnion, Pipeline
  from sklearn.tree     import DecisionTreeRegressor
  return Pipeline(steps=[
    ('dtr',EstimatorDecisiontree())
  ])

# COMMAND ----------

get_DecisionTree_pipeline() \
  .fit  (trn_coal_tfidf_fea_pdf, trn_coal_tfidf_tgt_ser) \
  .score(tst_coal_tfidf_fea_pdf, tst_coal_tfidf_tgt_ser)

# COMMAND ----------

get_DecisionTree_pipeline() \
  .fit  (trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser) \
  .score(tst_coal_cnt_fea_pdf, tst_coal_cnt_tgt_ser)

# COMMAND ----------

get_DecisionTree_pipeline() \
  .fit  (trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser) \
  .score(tst_ore_tfidf_fea_pdf, tst_ore_tfidf_tgt_ser)

# COMMAND ----------

get_DecisionTree_pipeline() \
  .fit  (trn_ore_cnt_fea_pdf, trn_ore_cnt_tgt_ser) \
  .score(tst_ore_cnt_fea_pdf, tst_ore_cnt_tgt_ser)

# COMMAND ----------

# MAGIC %md The scores are as following: 0.123, 0.246, 0.081, 0.100. It can be concluded that
# MAGIC the decision tree works best with `bci_coal_pdf` dataset containing features extracted using the CountVectorizer method.

# COMMAND ----------

# MAGIC %md ### Random Forest

# COMMAND ----------

# MAGIC %md Random forests, also known as random decision forests, are a popular ensemble method that can be used to build predictive models for both classification and regression problems. Ensemble methods use multiple learning models to gain better predictive results — in the case of a random forest, the model creates an entire forest of random uncorrelated decision trees to arrive at the best possible answer.

# COMMAND ----------

def get_RandomForest_pipeline():
  from sklearn.pipeline import FeatureUnion, Pipeline
  from sklearn.tree     import DecisionTreeRegressor
  return Pipeline(steps=[
    ('rf',EstimatorRandomForest())
  ])

# COMMAND ----------

# MAGIC %md Build `get_RandomForest_pipeline` function which returns a pipeline with the estimator [EstimatorRandomForest](https://bentley.cloud.databricks.com/#notebook/1607089/command/1607114), which has the base class `RandomForestRegressor` with the default parameters `n_estimators=5`, and `max_leaf_nodes=10`. Call the function and fit the pipeline with
# MAGIC 4 different datasets then get the R score on the relative test datasets. 

# COMMAND ----------

get_RandomForest_pipeline() \
  .fit  (trn_coal_tfidf_fea_pdf, trn_coal_tfidf_tgt_ser) \
  .score(tst_coal_tfidf_fea_pdf, tst_coal_tfidf_tgt_ser)

# COMMAND ----------

get_RandomForest_pipeline() \
  .fit  (trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser) \
  .score(tst_coal_cnt_fea_pdf, tst_coal_cnt_tgt_ser)

# COMMAND ----------

get_RandomForest_pipeline() \
  .fit  (trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser) \
  .score(tst_ore_tfidf_fea_pdf, tst_ore_tfidf_tgt_ser)

# COMMAND ----------

get_RandomForest_pipeline() \
  .fit  (trn_ore_cnt_fea_pdf, trn_ore_cnt_tgt_ser) \
  .score(tst_ore_cnt_fea_pdf, tst_ore_cnt_tgt_ser)

# COMMAND ----------

# MAGIC %md The scores are as following: 0.566, 0.639, 0.653, 0.579. It can be concluded that the decision tree works best with the dataset created from the merged `bci_ironore_pdf` features and `tfidfVectorizer` features.

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

# MAGIC %md  In this notebook different estimator pipelines were built and tested with their default parameters on the four different datasets created in Preprocessing pipeline.
# MAGIC Here's the score rank of all 7 models with the best outcome:
# MAGIC 
# MAGIC - Lasso Regression: 0.722 
# MAGIC - Random Forest: 0.691
# MAGIC - Lasso with PCA: 0.572
# MAGIC - Decision Tree: 0.246
# MAGIC - Ridge Regression: -0.547
# MAGIC - Elastic Net: - 1.602
# MAGIC - SVR: -2.881

# COMMAND ----------

# MAGIC %md The best Pipeline with a high `mean_test_score` of the R-square values is **Lasso Regression** with default parameters of `alpha=1` and `normalize='True'`. The scores of the model fitted and predicted on the train test dataset created from `bci_coal_pdf` dataset with features created using the `CounVectorizer` method and the `bci_ironore_pdf` dataset with additional features created using the `TfidfVectorizer` method are as below. <table>
# MAGIC     <tr>
# MAGIC         <td>`bci_coal_pdf`</td><td>`bci_ironore_pdf`</td><td>`CountVectorizer`</td><td>`TfidfVectorizer`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>Yes</td><td></td><td>Yes</td><td></td><td>72.21%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC     <tr>
# MAGIC         <td></td><td>Yes</td><td></td><td>Yes</td><td>71.80%</td>
# MAGIC     </tr>  
# MAGIC </table>
# MAGIC Hence in the Investigation models where gridsearch will be performed with different hyperparameters, these two combination of train-test datasets will be used to train and model and get the best score of each model and then select the best model with its hyperparameters.