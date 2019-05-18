# Databricks notebook source
# MAGIC %md #MA707 Report - Investigation (spring 2019, DataHeroes)

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC After doing all the pre-processing to find the best train test datasets and then fitting in to all the estimator pipelines with their default parameters, the scores were compared for all models with all 4 datasets. The two train-test datasets which gave the best scores for most of the estimator pipelines will then be used in this notebook where the models are investigated using Gridsearch with diferent hyperparameter settings and different combinations of estimator pipelines defined in the [Estimator Pipeline notebook](https://bentley.cloud.databricks.com/#notebook/1607450/command/1607451)

# COMMAND ----------

# MAGIC %md ## Contents
# MAGIC 1. Setup
# MAGIC 2. Hyperparameter Tuning
# MAGIC 3. Model evaluation and selection
# MAGIC 4. Summary

# COMMAND ----------

# MAGIC %md ## 1. Setup

# COMMAND ----------

# MAGIC %run "./3. Estimator pipelines"

# COMMAND ----------

from sklearn.pipeline        import FeatureUnion, Pipeline
from sklearn.linear_model    import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm             import SVR
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.decomposition   import PCA
from spark_sklearn           import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import make_scorer, mean_absolute_error, r2_score


# COMMAND ----------

# MAGIC %md ##2. Hyperparameter Tuning 

# COMMAND ----------

# MAGIC %md ##Model 1: Lasso Regression

# COMMAND ----------

# MAGIC %md Use GridSearchCV to get the best hyper parameters for Lasso Regression model.The hyperparameters are:
# MAGIC - `normalize`. True or False.The regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm or by their standard deviations. 
# MAGIC - `alpha`. It represents the regularization strength; Regularization improves the conditioning of the problem and reduces the variance of the estimates. Here we chose a range (0.001, 1000).
# MAGIC 
# MAGIC Then cross validation is defined as 5 time series splits which means it will train the model on combination of 4 subsets created from the training datset and validate the trained model on one subset. And the scoring method is R square which is a statistical measure of how close the data are to the fitted regression line. Then fit the gridsearchcv with features and target datasets

# COMMAND ----------

from spark_sklearn import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import make_scorer, mean_absolute_error, r2_score
lasso_run = \
GridSearchCV(sc,
  estimator=get_lasso_pipeline(),
  param_grid={'lso__normalize':[True,False],
              'lso__alpha'    :[10.0**n for n in range(-3,4)]},
  cv=TimeSeriesSplit(n_splits=5),
  scoring=make_scorer(r2_score),
  return_train_score=False,
  n_jobs=-1 
) 

lasso_run.fit(trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser)

display_pdf(est_grid_results_pdf(lasso_run,
                                 est_tag='lasso'))

# COMMAND ----------

lasso_run.fit(trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser)

display_pdf(est_grid_results_pdf(lasso_run,
                                 est_tag='lasso'))

# COMMAND ----------

# MAGIC %md The scores for the model with the best hyperparameters for the model trained and validated with the two datasets are:<table>
# MAGIC     <tr>
# MAGIC         <td>feature </td><td>target</td><td>`normalize`</td><td>`alpha`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_coal_cnt_fea_pdf`</td><td>`trn_coal_cnt_tgt_ser`</td><td>false</td><td>100</td><td>54.66%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_ore_tfidf_fea_pdf`</td><td>`trn_ore_tfidf_tgt_ser`</td><td>false</td><td>10</td><td>60.67%</td>
# MAGIC     </tr>
# MAGIC     
# MAGIC </table>

# COMMAND ----------

# MAGIC %md
# MAGIC - From the results it can be seen that when the hyperparameter `normalize` which normalizes the values between 0.0 and 1.0 is `false`, it consistantly produces the best predictions. 
# MAGIC - The best model for prediction using the lasso regression is when `normalize='False'` and `alpha` is 10 from a range of (0.001, 1000). The R-sqare value using these hyperparameter is 60.67%, that is 60.67% of the variance in the response target variable can be explained using this model.

# COMMAND ----------

# MAGIC %md ## Model 2: Lasso with PCA

# COMMAND ----------

# MAGIC %md Use GridSearchCV to get the best hyper parameters for Lasso Regression model with PCA.The hyperparameters tested are:
# MAGIC - `normalize`: `True` or `False`.The regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm or by their standard deviations. 
# MAGIC - `alpha`: It represents the regularization strength; Regularization improves the conditioning of the problem and reduces the variance of the estimates. Here we chose a range (0.001, 1000).
# MAGIC - `n_components`: It indicates how many principle components are created with the existing features. Principle components explain the variabilities of features.
# MAGIC 
# MAGIC Then cross validation is done using 5 time series splits and then the `mean_test_scores` for all the cross-validated tests are recorded. Then fit the GridsearchCV with features and target datasets

# COMMAND ----------

from spark_sklearn import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import make_scorer, mean_absolute_error, r2_score, mean_squared_error
pca_lasso_run = \
GridSearchCV(sc,
  estimator=get_pca_lasso_pipeline(),
  param_grid={'pca__n_components': [10**n for n in [2,3,4]],
              'lso__normalize':[True,False],
              'lso__alpha'    :[10.0**n for n in range(-3,4)]},
  cv=TimeSeriesSplit(n_splits=5),
  scoring=make_scorer(r2_score),
  return_train_score=False,
  n_jobs=-1 
) 

pca_lasso_run.fit(trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser)

display_pdf(est_grid_results_pdf(pca_lasso_run,
                                 est_tag='pca-lasso'))

# COMMAND ----------

pca_lasso_run.fit(trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser)

display_pdf(est_grid_results_pdf(pca_lasso_run,
                                 est_tag='pca-lasso-ironore'))

# COMMAND ----------

# MAGIC %md From the below test results we can get the best parameters:<table>
# MAGIC     <tr>
# MAGIC         <td>feature </td><td>target</td><td>`normalize`</td><td>`alpha`</td><td>`pca_n_components`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_coal_cnt_fea_pdf`</td><td>`trn_coal_cnt_tgt_ser`</td><td>`true`</td><td>10</td><td>10000</td><td>58.18%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_ore_tfidf_fea_pdf`</td><td>`trn_ore_tfidf_tgt_ser`</td><td>`false`</td><td>0.01</td><td>100</td><td>63.43%</td>
# MAGIC     </tr>
# MAGIC     
# MAGIC </table>

# COMMAND ----------

# MAGIC %md
# MAGIC - From the GridsearchCV results with the two training datasets, it can be seen that with the `trn_coal_cnt_` datset, when `alpha=10` and normalization is performed `normalize='True'`, it consistently produces better R-square values in the range of 58%. Whereas for the `trn_ore_tfidf_` dataframe the result is consistent with an alpha value in the range of 0.001 to 1. In this case the pca component number is consistent at 100.
# MAGIC - The best mean R-square value is 63.43% for the model trained with the `trn_ore_tfidf_` with hyperparameters `normalize='False'`, `alpha=0.01` and `pca_n_component=100`.
# MAGIC 
# MAGIC **Note** All the scores are better when compared with the default parameters ran in the estimator pipeline notebook.

# COMMAND ----------

# MAGIC %md ##Model 3: ElasticNet

# COMMAND ----------

# MAGIC %md Use GridSearchCV to get the best hyper parameters for the ElasticNet model.The hyperparameters are:
# MAGIC - `normalize`: True or False.The regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm or by their standard deviations. 
# MAGIC - `alpha`: It represents the regularization strength; Regularization improves the conditioning of the problem and reduces the variance of the estimates. Here we chose a range (0.001, 1000).
# MAGIC 
# MAGIC Then cross validation is defined as 5 time series splits. Then fit the gridsearchsv with features and target datasets

# COMMAND ----------

from spark_sklearn import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import make_scorer, mean_absolute_error, r2_score
elasticnet_run = \
GridSearchCV(sc,
  estimator=get_elasticnet_pipeline(),
  param_grid={'ela__normalize':[True,False],
              'ela__alpha'    :[10.0**n for n in range(-3,4)]},
  cv=TimeSeriesSplit(n_splits=5),
  scoring=make_scorer(r2_score),
  return_train_score=False,
  n_jobs=-1 
) 

elasticnet_run.fit(trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser)

display_pdf(est_grid_results_pdf(elasticnet_run,
                                 est_tag='elasticnet'))

# COMMAND ----------

elasticnet_run.fit(trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser)

display_pdf(est_grid_results_pdf(elasticnet_run,
                                 est_tag='elasticnet-ironore'))

# COMMAND ----------

# MAGIC %md From the below test results we can get the best hyperparameters:<table>
# MAGIC     <tr>
# MAGIC         <td>feature </td><td>target</td><td>`normalize`</td><td>`alpha`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_coal_cnt_fea_pdf`</td><td>`trn_coal_cnt_tgt_ser`</td><td>`true`</td><td>1000</td><td>48.52%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_ore_tfidf_fea_pdf`</td><td>`trn_ore_tfidf_tgt_ser`</td><td>`false`</td><td>0.01</td><td>60.79%</td>
# MAGIC     </tr>
# MAGIC     
# MAGIC </table>

# COMMAND ----------

# MAGIC %md
# MAGIC - From both the GridsearchCV output, it can be seen that the best scores for both the models is consistent when the parameter `normalize` is `false`. It consistently produces the `mean_test_score` of above 50% for `alpha` values in the range of 0.001 to 1 for the model trained on the `_ore_tfidf_` dataset.
# MAGIC - When normalize is false, and alpha is 0.01, it gives the best R square value of 60.79%.

# COMMAND ----------

# MAGIC %md ##Model 4: ElasticNet with PCA

# COMMAND ----------

# MAGIC %md Similarly run the GridSearchCV on the `get_pca_elasticnet_pipeline()` which runs the object `pca` which is the class FeatureSelectionPCA() and then fitted into the ElasticNet estimator to get the best hyper parameters for the model.The hyperparameters tested are:
# MAGIC - `normalize`: `True` or `False`.The regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm or by their standard deviations. 
# MAGIC - `alpha`: It represents the regularization strength; Regularization improves the conditioning of the problem and reduces the variance of the estimates. Here we chose a range (0.001, 1000).
# MAGIC - `n_components`: It indicates how many principle components are created with the existing features. Principle components explain the variabilities of features. Here we define the range as `[10^2, 10^4]`
# MAGIC 
# MAGIC The model is cross validated using the 5 time series splits and the `mean_test_scores` of R-square values for the cross-validation are compard and ranked accordingly. Then fit the GridsearchCV with the two different features and target datasets

# COMMAND ----------

from spark_sklearn import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import make_scorer, mean_absolute_error, r2_score
pca_elasticnet_run = \
GridSearchCV(sc,
  estimator=get_pca_elasticnet_pipeline(),
  param_grid={'pca__n_components': [10**n for n in [2,3,4]],
              'ela__normalize':[True,False],
              'ela__alpha'    :[10.0**n for n in range(-3,4)]},
  cv=TimeSeriesSplit(n_splits=5),
  scoring=make_scorer(r2_score),
  return_train_score=False,
  n_jobs=-1 
) 

pca_elasticnet_run.fit(trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser)

display_pdf(est_grid_results_pdf(pca_elasticnet_run,
                                 est_tag='pca-elasticnet'))

# COMMAND ----------

pca_elasticnet_run.fit(trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser)

display_pdf(est_grid_results_pdf(pca_elasticnet_run,
                                 est_tag='pca-elasticnet-ironore'))

# COMMAND ----------

# MAGIC %md From the below test results we can get the best parameters:<table>
# MAGIC     <tr>
# MAGIC         <td>feature </td><td>target</td><td>`normalize`</td><td>`alpha`</td><td>`pca_n_components`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_coal_cnt_fea_pdf`</td><td>`trn_coal_cnt_tgt_ser`</td><td>`true`</td><td>0.001</td><td>100</td><td>69.18%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_ore_tfidf_fea_pdf`</td><td>`trn_ore_tfidf_tgt_ser`</td><td>`true`</td><td>0.01</td><td>100</td><td>63.61%</td>
# MAGIC     </tr>
# MAGIC     
# MAGIC </table>

# COMMAND ----------

# MAGIC %md 
# MAGIC - From the `mean_test_score` results from the GridsearchCV on the two different datasets, the most consistent hyperparameter with better scores for both model is when `normalize='False'`and when `pca_n_component=100`. Although the best score for the two models is when this hyperparameter `normalize` is `'True'` fand the highest at 69.18%, it consistently gives better scores for different `alpha` values as well as `pca_n_components` when the values are not normalized.
# MAGIC - The best `mean_test_score` for the two models is 69.18% with the `_coal_cnt_` dataset pre-processed with the CountVectorizer method with the hyperparameters `normalize='True'`, `alpha=0.001` and `pca_n_component=100`.

# COMMAND ----------

# MAGIC %md ##Model 5: Ridge Regression

# COMMAND ----------

# MAGIC %md Using similar hyperparameter values as for Lasso regression for `normalize` and `alpha`, run the GridsearchCV for the Ridge Regression model with different solvers `['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']`

# COMMAND ----------

# MAGIC %md **Note:** Solver to use in the computational routines:
# MAGIC 
# MAGIC - `auto` chooses the solver automatically based on the type of data.
# MAGIC - `svd` uses a Singular Value Decomposition of X to compute the Ridge coefficients. More stable for singular matrices than ‘cholesky’.
# MAGIC - `cholesky` uses the standard scipy.linalg.solve function to obtain a closed-form solution.
# MAGIC - `sparse_cg` uses the conjugate gradient solver as found in scipy.sparse.linalg.cg. As an iterative algorithm, this solver is more appropriate than `cholesky` for large-scale data (possibility to set tol and max_iter).
# MAGIC - `lsqr` uses the dedicated regularized least-squares routine scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative procedure.
# MAGIC - `sag` uses a Stochastic Average Gradient descent, and ‘saga’ uses its improved, unbiased version named SAGA. Both methods also use an iterative procedure, and are often faster than other solvers when both n_samples and n_features are large. Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.%

# COMMAND ----------

from spark_sklearn import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import make_scorer, mean_absolute_error, r2_score
ridge_run = \
GridSearchCV(sc,
  estimator=get_ridge_pipeline(),
  param_grid={'rdg__normalize':[True,False],
              'rdg__alpha'    :[10.0**n for n in range(-3,4)],
              'rdg__solver'   :['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']},
  cv=TimeSeriesSplit(n_splits=5),
  scoring=make_scorer(r2_score),
  return_train_score=False,
  n_jobs=-1 
) 

ridge_run.fit(trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser)

display_pdf(est_grid_results_pdf(ridge_run,
                                 est_tag='ridge'))

# COMMAND ----------

ridge_run.fit(trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser)

display_pdf(est_grid_results_pdf(ridge_run,
                                 est_tag='ridge-ironore'))

# COMMAND ----------

# MAGIC %md The top `mean_test_score` values for the two models tested with the hyperparameters as defined above are as below:<table>
# MAGIC     <tr>
# MAGIC         <td>feature </td><td>target</td><td>`normalize`</td><td>`alpha`</td><td>`solver`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_coal_cnt_fea_pdf`</td><td>`trn_coal_cnt_tgt_ser`</td><td>`False`</td><td>1</td><td>`saga`</td><td>51.02%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_ore_tfidf_fea_pdf`</td><td>`trn_ore_tfidf_tgt_ser`</td><td>`False`</td><td>1</td><td>`sparse_cg`</td><td>60.24%</td>
# MAGIC     </tr>
# MAGIC     
# MAGIC </table>
# MAGIC 
# MAGIC Comparing the test scores, it can be observed that when the hyperparameter `normalize` is `'False'` and `alpha=1`, it consistently produces the best predictions for differnt solvers. 
# MAGIC The best score from the gridsearchCV is 60.24% with the hyperparameters `alpha=1`, `normalize='False'` and the `sparse_cg` solver is used in the model.

# COMMAND ----------

# MAGIC %md ##Model 6: Ridge Regression with PCA

# COMMAND ----------

# MAGIC %md The same Ridge Regression model is then re-run with the same hyperparameters but with an additional feature reduction method for reducing the dimensionality of data. The `FeatureSelectionPCA` method is run in the pipeline which has a base class `PCA` and the hyperparameter `pca_n_component` is set as [10, 20,30].
# MAGIC 
# MAGIC So the model runs the Ridge Regression on the dataset with number of features reduced by the `FeatureSelectionPCA` method to 10, 20 and 30 using the Principal Component Analysis. This model is then cross-validated using a 5 time-series split of the training dataset and the R-square values are averaged out for the cross-validation and then are compared for all other hyperparameters.

# COMMAND ----------

pca_ridge_run = \
GridSearchCV(sc,
            estimator=Pipeline(steps=[('pca',FeatureSelectionPCA()),
                                      ('rdg',EstimatorRidge())
                                     ]),
            param_grid={'rdg__normalize'   :[True, False],
                        'rdg__alpha'       :[10.0**n for n in [-3,0,3]],
                        'rdg__solver'      :['saga'],
                        'pca__n_components':[10*n for n in [1, 2, 3]]
            },
 cv=TimeSeriesSplit(n_splits=5),
 scoring=make_scorer(r2_score),
 return_train_score=False,
 n_jobs=-1
)
pca_ridge_run \
 .fit(trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser)
display_pdf(est_grid_results_pdf(pca_ridge_run,
                                est_tag='pca-ridge'))

# COMMAND ----------

pca_ridge_run.fit(trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser)

display_pdf(est_grid_results_pdf(pca_ridge_run,
                                 est_tag='pca-ridge-ironore'))

# COMMAND ----------

# MAGIC %md The best `mean_test_score` with the hyperparameters for the models are:<table>
# MAGIC     <tr>
# MAGIC         <td>feature </td><td>target</td><td>`normalize`</td><td>`alpha`</td><td>`solver`</td><td>`pca_n_components`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_coal_cnt_fea_pdf`</td><td>`trn_coal_cnt_tgt_ser`</td><td>`False`</td><td>1</td><td>`saga`</td><td>20</td><td>50.77%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_ore_tfidf_fea_pdf`</td><td>`trn_ore_tfidf_tgt_ser`</td><td>`True`</td><td>0.001</td><td>`saga`</td><td>30</td><td>55.62%</td>
# MAGIC     </tr>
# MAGIC     
# MAGIC </table>
# MAGIC 
# MAGIC - The hyperparameter which produced the most consistent test score is the `rdg_solver='saga'` followed by `alpha=0.001` with scores above 50%. 
# MAGIC - When `normalize` is `true`, and `alpha` is 0.001 with solver as `saga` with a dimensionality of 30 (`pca_n_component=30`), it gives the best R square value of 55.62%.

# COMMAND ----------

# MAGIC %md ### MODEL 7: Decision Tree

# COMMAND ----------

# MAGIC %md Use Decision tree regressor with grid search to find out the best tree depth and maximum leaf nodes. The estimator pipeline used in the gridsearch is `decision_tree_regressor`, and the parameters are as below:
# MAGIC - tree depth (`max_depth`). The range is from 1 to 10. The deeper the tree, the more complex the decision rules and the fitter the model.
# MAGIC - `max_leaf_nodes`. It represents the max leaf nodes that the tree can grow. Here it is determined by the list of numbers [5, 10, 15, 20, 25]. The tree will stop growing once the maximum number is reached.

# COMMAND ----------

dtr_run = \
GridSearchCV(sc,
             estimator=get_DecisionTree_pipeline(),
             param_grid={'dtr__max_depth'        : [1,2,3,4,5,6,7,8,9,10],
                         'dtr__max_leaf_nodes': [5, 10, 15, 20, 25]
                        },
             cv=TimeSeriesSplit(n_splits=5),
             scoring=make_scorer(r2_score),
             return_train_score=False,
             n_jobs=-1 
            ) 
dtr_run \
  .fit(trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser)
display_pdf(est_grid_results_pdf(dtr_run,
                                 est_tag='dtr'))


# COMMAND ----------

dtr_run.fit(trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser)

display_pdf(est_grid_results_pdf(dtr_run,
                                 est_tag='dtr-ironore'))

# COMMAND ----------

# MAGIC %md The test results from the gridsearch with the `decision_tree_regressor` pipeline are:<table>
# MAGIC     <tr>
# MAGIC         <td>feature </td><td>target</td><td>`max_depth`</td><td>`max_leaf_nodes`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_coal_cnt_fea_pdf`</td><td>`trn_coal_cnt_tgt_ser`</td><td>10</td><td>10</td><td>53.93%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_ore_tfidf_fea_pdf`</td><td>`trn_ore_tfidf_tgt_ser`</td><td>8</td><td>10</td><td>49.69%</td>
# MAGIC     </tr>
# MAGIC     
# MAGIC </table>
# MAGIC 
# MAGIC - From the GridseachCV results, it can be clearly seen that none of the hyperparameters are consistent in producing the better scores.
# MAGIC - The best result of an R-square value of 53.93% is produced with the hyperparameters `max_depth` is 10 and `max_leaf_nodes` as 10 with the training dataset produced using the `TfidfVectorizer` method on the `bci_ironore_pdf` dataset.

# COMMAND ----------

# MAGIC %md ### MODEL 8: Random Forest

# COMMAND ----------

# MAGIC %md A random forest is simply a collection of decision trees whose results are aggregated into one final result. Their ability to limit overfitting without substantially increasing error due to bias is why they are such powerful models. One way Random Forests reduce variance is by training on different samples of the data.
# MAGIC 
# MAGIC Use Random Forest regressor with grid search to find out the best tree numbers and leaf nodes. Pipeline is random forest regressor, param_grid includes:
# MAGIC - `n_estimators`: The list of number of trees in forest. Here it is determined by the list of numbers [5,10,20]. 
# MAGIC - `max leaf nodes`: It represents the maximum leaf nodes that the trees can grow. Here it is determined by the list of numbers [5, 10, 15, 20, 25]. The trees will stop growing once the maximum number is reached
# MAGIC 
# MAGIC The the cross-validation scores for all the models with different combinations of hyperparameters are then averaged out to get the `mean_test_score` and ranked accordingly.

# COMMAND ----------

rf_run = \
GridSearchCV(sc,
             estimator=get_RandomForest_pipeline(),
             param_grid={'rf__n_estimators'  : [5, 10, 20],
                         'rf__max_leaf_nodes': [5, 10, 15, 20, 25]
                        },
             cv=TimeSeriesSplit(n_splits=5),
             scoring=make_scorer(r2_score),
             return_train_score=False,
             n_jobs=-1 
            ) 
rf_run \
  .fit(trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser)
display_pdf(est_grid_results_pdf(rf_run,
                                 est_tag='rf'))

# COMMAND ----------

rf_run.fit(trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser)

display_pdf(est_grid_results_pdf(rf_run,
                                 est_tag='rf-ironore'))

# COMMAND ----------

# MAGIC %md The following test results we can get the best hyperparameters for the Random Forest regression model:<table>
# MAGIC     <tr>
# MAGIC         <td>feature </td><td>target</td><td>`n_estimators`</td><td>`max_leaf_nodes`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_coal_cnt_fea_pdf`</td><td>`trn_coal_cnt_tgt_ser`</td><td>20</td><td>10</td><td>58.76%</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_ore_tfidf_fea_pdf`</td><td>`trn_ore_tfidf_tgt_ser`</td><td>5</td><td>10</td><td>61.46%</td>
# MAGIC     </tr>
# MAGIC     
# MAGIC </table>
# MAGIC 
# MAGIC - As seen from the above test scores, random forest produces improved `mean_test_scores` than the Decision Tree Model. The most consistent hyperparameter with better test score is `max_leaf_nodes=10`.
# MAGIC - And the best score for the model is when `n_estimators` is 5, and `max_leaf_nodes` is 10, it gives the best mean R square value of 61.46%.

# COMMAND ----------

# MAGIC %md ## MODEL EVALUATION AND SELECTION

# COMMAND ----------

# MAGIC %md Using GridsearchCV all the different regression models were investigated with different hyperparameters and tuned to get the best scores for the model. Some of the models improved their r-square values after tuning with different hyperparameters while some of the scores were reduced. 
# MAGIC For example the estimator `ElasticLasso` with the default parameters of `alpha=1`  and `normalize=True` had the highest `mean_test_score` with and R-square value of 72.21% when compared to other models with default values, whereas after the hyperparameter during using GridsearchCV, this reduced to 60.79%.
# MAGIC 
# MAGIC Similarly the ElasticNet model which had a negative mean test score improved after hyperparameter tuning and gave the best score of 69.18%. 
# MAGIC 
# MAGIC After testing with different hyperparameter values, here's the score rank of all 8 models with the best outcome:
# MAGIC 
# MAGIC - Model 1: Elastic Net with PCA: 69.18%
# MAGIC - Model 2: Lasso with PCA: 63.43%
# MAGIC - Model 3: Random Forest: 61.46%
# MAGIC - Model 4: Elastic Net: 60.79%
# MAGIC - Model 5: Lasso Regression: 60.67% 
# MAGIC - Model 6: Ridge Regression: 60.24%
# MAGIC - Model 7: Decision Tree: 53.93%
# MAGIC - Model 8: Ridge with PCA: 52.62%
# MAGIC 
# MAGIC Although PCA has several advantages, but the main drawback of PCA is that the decision about how many principal components to keep does not depend on the response variable. Consequently, some of the variables that is selected might not be strong predictors of the response, and some of the components that is dropped might be excellent predictors. It does not consider the response variable when deciding which principal components to drop. The decision to drop components is based only on the magnitude of the variance of the components.
# MAGIC 
# MAGIC Also it makes the independent variables selected less interpretable. Hence as the final models for prediction, the Random Forest model is also selected to compare the prediction score and their accuracies with the ElasticNet model.

# COMMAND ----------

# MAGIC %md From the study on the gridsearch scores for various regression models, the best model to predict the `bci_5tc` price is **Elastic Net with PCA** using the features from the `bci_pdf` and additional features extracted from the `coal_pdf` using `CountVectorizer` method. This model combines the penalties of both ridge regression and lasso to take into account of overfitting and to get the best of both worlds.
# MAGIC 
# MAGIC The best hyperparameters for the two models to be compared and get the test score for predicting are:
# MAGIC 
# MAGIC   - **Model 1: ElasticNet with PCA**<table>
# MAGIC     <tr>
# MAGIC         <td>Feature</td><td>Target</td><td>**`Normalize`**</td><td>**`Alpha`**</td><td>**`Pca_n_components`**</td><td>R Square</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_coal_cnt_fea_pdf`</td><td>`trn_coal_cnt_tgt_ser`</td><td>`true`</td><td>0.001</td><td>100</td><td>69.18%</td>
# MAGIC     </tr>
# MAGIC     
# MAGIC </table>
# MAGIC 
# MAGIC   - **Model 2: Random Forest**<table>
# MAGIC     <tr>
# MAGIC         <td>Feature</td><td>Target</td><td>`n_estimators`</td><td>`max_leaf_nodes`</td><td>`R Square`</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td>`trn_ore_tfidf_fea_pdf`</td><td>`trn_ore_tfidf_tgt_ser`</td><td>5</td><td>10</td><td>61.46%</td>
# MAGIC     </tr>
# MAGIC     
# MAGIC </table>  

# COMMAND ----------

# MAGIC %md **Model 1: ElasticNet with PCA**
# MAGIC The function for the PCA ElasticNet pipeline model with the hyperparameters selected from the GridsearchCV results is defined and this model is tested on the test dataset.

# COMMAND ----------

# MAGIC %python 
# MAGIC def tst_pca_elasticnet_pipeline():
# MAGIC   from sklearn.pipeline import FeatureUnion, Pipeline
# MAGIC   from sklearn.linear_model import LinearRegression, Ridge
# MAGIC   return Pipeline(steps=[
# MAGIC     ('pca', FeatureSelectionPCA(n_components=100)),
# MAGIC     ('ela', EstimatorElasticNet(normalize=True, alpha=0.001))
# MAGIC   ])

# COMMAND ----------

tst_pca_elasticnet_pipeline() \
  .fit  (trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser) \
  .score(tst_coal_cnt_fea_pdf, tst_coal_cnt_tgt_ser)

# COMMAND ----------

# MAGIC %md The r-square value for the prediction is 57.8% which is a bit lower than the r-square value of 69% on the training dataset. 
# MAGIC 
# MAGIC The sklearn.metrics module implements several loss, score, and utility functions to measure regression performance. Metrics are used to evaluate a model by comparing the actual values with the predicted values produced by the model. Some of these metrics used to evaluate the regression model are: `mean_squared_error`, `mean_absolute_error`, `explained_variance_score`, `mean_squared_log_error`, `median_absolute_error` and `r2_score`. Inorder to compare the predicted value to the observed response value, the predicted `bci_5tc` values are calculated using `.predict()` to the fitted model.

# COMMAND ----------

predicted = \
tst_pca_elasticnet_pipeline() \
.fit  (trn_coal_cnt_fea_pdf, trn_coal_cnt_tgt_ser) \
.predict(tst_coal_cnt_fea_pdf)

observed = np.array(tst_coal_cnt_tgt_ser)


# COMMAND ----------

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score

print('Explained Variance Score : ' + str(explained_variance_score(observed,predicted)))
print('Mean Absolute Error      : ' + str(mean_absolute_error(observed,predicted)))
print('Mean Squared Error       : ' + str(mean_squared_error(observed,predicted)))
print('Mean Squared Log Error   : ' + str(mean_squared_log_error(observed,predicted)))
print('Median Absolute Error    : ' + str(median_absolute_error(observed,predicted)))
print('R Squared (R2)           : ' + str(r2_score(observed,predicted)))

# COMMAND ----------

# MAGIC %md **Model 2: Random Forest**
# MAGIC The random forest model is defined with the hyperparameters selected from the Gridsearch results, `n_estimators=5` and `max_leaf_nodes=10`. This model is then trained using the `bci_ironore_tfidf` training dataset and then tested with the testing dataset. The scores and metrics are calculated below. The R-square value for prediction is 66.05% which is better than the training score of 61%. 

# COMMAND ----------

from sklearn.tree     import DecisionTreeRegressor
rf = RandomForestRegressor(n_estimators=5, max_leaf_nodes=10)

rf.fit(trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser) \
.score(tst_ore_tfidf_fea_pdf, tst_ore_tfidf_tgt_ser)


# COMMAND ----------

rf_predicted = rf.fit  (trn_ore_tfidf_fea_pdf, trn_ore_tfidf_tgt_ser).predict(tst_ore_tfidf_fea_pdf)
rf_observed = np.array(tst_ore_tfidf_tgt_ser)

# COMMAND ----------

# MAGIC %md Measure the regression performance for the model and compare the two models.

# COMMAND ----------

print('Explained Variance Score : ' + str(explained_variance_score(rf_observed,rf_predicted)))
print('Mean Absolute Error      : ' + str(mean_absolute_error(rf_observed,rf_predicted)))
print('Mean Squared Error       : ' + str(mean_squared_error(rf_observed,rf_predicted)))
print('Mean Squared Log Error   : ' + str(mean_squared_log_error(rf_observed,rf_predicted)))
print('Median Absolute Error    : ' + str(median_absolute_error(rf_observed,rf_predicted)))
print('R Squared (R2)           : ' + str(r2_score(rf_observed,rf_predicted)))

# COMMAND ----------

# MAGIC %md When the metrics scores for the two model performance are compared, it is observed that all the error scores are less for Random Forest compared to that of the ElasticNet model. As the R-square value for the Random Forest model is better for the prediction on test sets with a score of 66%, it can be concluded to be the better fit model to predict the `bci_5tc` price. 
# MAGIC 
# MAGIC Also for random forest models, important features for the prediction model can also be calculated with their importance value. It is calculated using the `feature_importance_` method in sklearn. 
# MAGIC Feature importance is calculated as the decrease in node impurity weighted by the probability of reaching that node. The node probability can be calculated by the number of samples that reach the node, divided by the total number of samples. The higher the value the more important the feature.
# MAGIC From the results below, it is noticed that the variables `bci_`, `c5_` are the most important features to give an accurate prediction with the model.

# COMMAND ----------

feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = trn_ore_tfidf_fea_pdf.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances

# COMMAND ----------

# MAGIC %md From the above findings the Random Forest Model is selected as the best model for the prediction of the target variable `bci_5tc`. The features used for the model are the lagged version of all features from the time series dataset `bci_pdf` and the extracted features from the mining text dataset `iron_ore_pdf` using `TfidfVectorizer` method.

# COMMAND ----------

# MAGIC %md ##SUMMARY: 
# MAGIC In this report different machine learning classes and feature selection techniques are used. Several pipelines with different estimators and wrapper classes were used to design the regression models which were then investigated to get the best hyperparameters using GridsearchCV. The `mean_test_scores` for each model with different hyperparameters were compared for each model and ranked accordingly in the section **Hyperparameter Tuning**. Ultimately the best two models **ElasticNet with PCA and Random Forest models** were selected as the final models to predict using the test datasets and compare their results with the observed response variable in model evaluation.
# MAGIC 
# MAGIC Out of the two models with their tuned hyperparameters, Random forest has the best test score with lower error metrics and interpretable predictor variables which can be used to explain the model better. 