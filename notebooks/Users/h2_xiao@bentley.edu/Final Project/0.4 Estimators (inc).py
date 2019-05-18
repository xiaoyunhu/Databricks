# Databricks notebook source
# MAGIC %md This notebook includes wrapper classes for estimators.

# COMMAND ----------

# MAGIC %md ### Ridge Model Wrapper Class

# COMMAND ----------

# MAGIC %md Ridge regression is a regression technique for continuous value prediction. It is a simple techniques to reduce model complexity and prevent over-fitting which may result from simple linear regression. Using a ridge regression method helps reduce the multicollinearity of endogenous variables in models. Multicollinearity creates inaccurate estimates of the regression coefficients.
# MAGIC 
# MAGIC Define a class `EstimatorRidge()` which calls on the base class `Ridge` with all its parameters.

# COMMAND ----------

from sklearn.linear_model    import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.base              import BaseEstimator,TransformerMixin

class EstimatorRidge(Ridge):
    def __init__(self, normalize=True, alpha=1, solver='saga'
                ):
        self.normalize= normalize
        self.alpha = alpha
        self.solver = solver
        super().__init__(alpha=self.alpha, normalize=self.normalize, solver=self.solver)

    def fit(self, X, y, sample_weight=None):
         self.sample_weight = sample_weight
         return super().fit(X, y, sample_weight=self.sample_weight)
      
    def predict(self, X):
        return super().predict(X)

# COMMAND ----------

# MAGIC %md 
# MAGIC  In the class `EstimatorRidge` the init method defines the parameters which then inherits the base class `Ridge` via delegation using `super()` method. The self variable represents the instance of the object itself. 
# MAGIC  
# MAGIC The parameters defined in the class are:<br>
# MAGIC -  `alpha=1`: Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. <br>
# MAGIC - `normalize=True`: The regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm or by their standard deviations. This parameter is ignored when `fit_intercept` in a regression model is set to `False`. 
# MAGIC - `solver='saga'`: Its an improved unbiased version of the Stochastic Average Gradient descent (`sag`) solver. Both methods also use an iterative procedure, and are often faster than other solvers when both n_samples and n_features are large. Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale.

# COMMAND ----------

# MAGIC %md ### Lasso Model Wrapper Class

# COMMAND ----------

# MAGIC %md Lasso Regression, another regularization method used to prevent overfitting. Lasso regression has a clear advantage over ridge regression. As ridge regression shrink coefficients towards zero, it can never reduce it to zero so, all features will be included in the model no matter how small the value of the coefficients. Lasso regression, is able to shrink coefficient to exactly zero, reducing the number of features and serve as a feature selection tools at the same time. This makes Lasso regression useful in cases with high dimension and helps with model interpretability.
# MAGIC 
# MAGIC Define a class `EstimatorLasso()` which calls on the base class `Lasso` with all its parameters.

# COMMAND ----------

from sklearn.linear_model    import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.base              import BaseEstimator,TransformerMixin

class EstimatorLasso(Lasso):
    def __init__(self, normalize=True, alpha=1
                ):
        self.normalize= normalize
        self.alpha = alpha
        super().__init__(alpha=self.alpha, normalize=self.normalize)

    def fit(self, X, y):
         return super().fit(X, y)
      
    def predict(self, X):
        return super().predict(X)

# COMMAND ----------

# MAGIC %md 
# MAGIC  In the class `EstimatorLasso` the init method defines the parameters which then inherits the base class `Lasso` via delegation using `super()` method. The self variable represents the instance of the object itself. 
# MAGIC  
# MAGIC The parameters defined in the class are:<br>
# MAGIC -  `alpha=1`: Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. <br>
# MAGIC - `normalize=True`: The regressors X will be normalized before regression by subtracting the mean and dividing by the l1-norm or by their standard deviations. This parameter is ignored when `fit_intercept` in a regression model is set to `False`. 

# COMMAND ----------

# MAGIC %md ### ElasticNet Model Wrapper Class

# COMMAND ----------

# MAGIC %md  ElasticNet is a third commonly used model of regression which incorporates penalties from both Ridge and Lasso models. Addition to setting and choosing a lambda value elastic net also allows us to tune the alpha parameter where 𝞪 = 0 corresponds to ridge and 𝞪 = 1 to lasso. Therefore we can choose an alpha value between 0 and 1 to optimize the elastic net. Effectively this will shrink some coefficients and set some to 0 for sparse selection.
# MAGIC 
# MAGIC Define a class `EstimatorElasticNet()` which calls on the base class `ElasticNet` with all its parameters.

# COMMAND ----------

from sklearn.linear_model    import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.base              import BaseEstimator,TransformerMixin

class EstimatorElasticNet(ElasticNet):
    def __init__(self, normalize=True, alpha=0.5
                ):
        self.normalize= normalize
        self.alpha = alpha
        super().__init__(alpha=self.alpha, normalize=self.normalize)

    def fit(self, X, y):
         return super().fit(X, y)
      
    def predict(self, X):
        return super().predict(X)

# COMMAND ----------

# MAGIC %md 
# MAGIC  In the class `EstimatorElasticNet` the init method defines the parameters which then inherits the base class `ElasticNet` via delegation using `super()` method. The self variable represents the instance of the object itself. 
# MAGIC  
# MAGIC The parameters defined in the class are:<br>
# MAGIC -  `alpha=0.5`: Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. <br>
# MAGIC - `normalize=True`: The regressors X will be normalized before regression by subtracting the mean and dividing by the l1-l2-norm or by their standard deviations. This parameter is ignored when `fit_intercept` in a regression model is set to `False`. 

# COMMAND ----------

# MAGIC %md ### SVR Model Wrapper Class

# COMMAND ----------

# MAGIC %md Support Vector Regression (SVR) uses the same principles as the SVM for classification, with only a few minor differences. In simple regression we try to minimise the error rate. While in SVR we try to fit the error within a certain threshold.  Objective with SVR is to basically consider the points that are within the boundary line, so best fit line is the line hyperplane that has maximum number of points.
# MAGIC 
# MAGIC Define a class `EstimatorSVR()` which calls on the base class `ElasticNet` with all its parameters.

# COMMAND ----------

from sklearn.svm               import SVR
from sklearn.base              import BaseEstimator,TransformerMixin

class EstimatorSVR(SVR):
    def __init__(self, kernel='rbf', gamma='auto', C=100
                ):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        super().__init__(kernel=self.kernel, gamma=self.gamma, C=self.C)

    def fit(self, X, y, sample_weight=None):
         self.sample_weight = sample_weight
         return super().fit(X, y, sample_weight=self.sample_weight)
      
    def predict(self, X):
        return super().predict(X)

# COMMAND ----------

# MAGIC %md 
# MAGIC  In the class `EstimatorSVR` the init method defines the parameters which then inherits the base class `SVR` via delegation using `super()` method. The self variable represents the instance of the object itself. 
# MAGIC  
# MAGIC The parameters defined in the class are:<br>
# MAGIC -  `kernel=rbf`: The function used to map a lower dimensional data into a higher dimensional data. <br>
# MAGIC - `gamma=gamma`: The regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm or by their standard deviations. This parameter is ignored when `fit_intercept` in a regression model is set to `False`. 

# COMMAND ----------

# MAGIC %md ###Model: K-nearest neighbors

# COMMAND ----------

# MAGIC %md **K-nearest neighbors** are used in cases where the data labels are continuous rather than discrete variables. The label assigned to a query point is computed based on the mean of the labels of its nearest neighbors. In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.
# MAGIC 
# MAGIC The basic nearest neighbors regression uses uniform weights: that is, each point in the local neighborhood contributes uniformly to the classification of a query point. Under some circumstances, it can be advantageous to weight points such that nearby points contribute more to the regression than faraway points. This can be accomplished through the weights keyword. The default value, `weights = 'uniform'`, assigns equal weights to all points. `weights = 'distance'` assigns weights proportional to the inverse of the distance from the query point. Alternatively, a user-defined function of the distance can be supplied, which will be used to compute the weights.

# COMMAND ----------

from sklearn.neighbors       import KNeighborsRegressor
class EstimatorKNREG(KNeighborsRegressor):
    def __init__(self, n_neighbors='n_neighbors', weights='weights', p=[10]
                ):
        self.n_neighbors= n_neighbors
        self.weights = weights
        self.p=p
        super().__init__(n_neighbors=self.n_neighbors, weights=self.weights, p=self.p)

    def fit(self, X, y, sample_weight=None):
         self.sample_weight = sample_weight
         return super().fit(X, y, sample_weight=self.sample_weight)
      
    def predict(self, X):
        return super().predict(X)

# COMMAND ----------

# MAGIC %md The parameters defined are <br>`p` : integer, optional (default = 2)
# MAGIC Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.<br>
# MAGIC `n_neighbors`: The number ofnearest neighbors contributing to the regression

# COMMAND ----------

# MAGIC %md ###Model: Decision Tree

# COMMAND ----------

# MAGIC %md A decision tree is a largely used non-parametric effective machine learning modeling technique for regression and classification problems. To find solutions a decision tree makes sequential, hierarchical decision about the outcomes variable based on the predictor data.
# MAGIC 
# MAGIC Hierarchical means the model is defined by a series of questions that lead to a class label or a value when applied to any observation. Once set up the model acts like a protocol in a series of “if this occurs then this occurs” conditions that produce a specific result from the input data.
# MAGIC 
# MAGIC A Non-parametric method means that there are no underlying assumptions about the distribution of the errors or the data. It basically means that the model is constructed based on the observed data.

# COMMAND ----------

from sklearn.tree            import DecisionTreeRegressor

class EstimatorDecisiontree(DecisionTreeRegressor):
    def __init__(self,max_depth=5, max_leaf_nodes=10):
        self.max_depth= max_depth
        self.max_leaf_nodes = max_leaf_nodes
        super().__init__(max_depth=self.max_depth,  max_leaf_nodes=self. max_leaf_nodes)

    def fit(self, X, y, sample_weight=None):
         self.sample_weight = sample_weight
         return super().fit(X, y, sample_weight=self.sample_weight)
      
    def predict(self, X):
        return super().predict(X)

# COMMAND ----------

# MAGIC %md ###Model: Random Forest

# COMMAND ----------

# MAGIC %md Random forests, also known as random decision forests, are a popular ensemble method that can be used to build predictive models for both classification and regression problems. Ensemble methods use multiple learning models to gain better predictive results — in the case of a random forest, the model creates an entire forest of random uncorrelated decision trees to arrive at the best possible answer.

# COMMAND ----------

from sklearn.ensemble        import RandomForestRegressor

class EstimatorRandomForest(RandomForestRegressor):
    def __init__(self, n_estimators=5, max_leaf_nodes=10):
        self.n_estimators= n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        super().__init__(n_estimators=self.n_estimators,  max_leaf_nodes=self. max_leaf_nodes)

    def fit(self, X, y, sample_weight=None):
         self.sample_weight = sample_weight
         return super().fit(X, y, sample_weight=self.sample_weight)
      
    def predict(self, X):
        return super().predict(X)
      