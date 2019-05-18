# Databricks notebook source
# MAGIC %md This notebook includes wrapper classes for feature selection. 
# MAGIC 
# MAGIC The classes below may need to be rewritten using inheritance and the `super` function. 

# COMMAND ----------

import pandas as pd
import numpy  as np

# COMMAND ----------

import numpy as np
def variance_scorer (X,y=None):
  return tuple([X.apply(np.var), 
                None
               ]
              )

# COMMAND ----------

# MAGIC %md ## 2. `SelectKBestDF`

# COMMAND ----------

from sklearn.feature_selection import SelectKBest
from sklearn.base              import BaseEstimator,TransformerMixin
from sklearn.feature_selection import chi2

class SelectKBestDF(BaseEstimator,TransformerMixin):

  def __init__(self, score_func=chi2, k=10):
    self.score_func = score_func
    self.k          = k

  def fit(self, X, y=None):
    (self.scores_,
     self.pvalues_
    ) = self.score_func(X,y)
    var_nam           = X.columns
    var_ndx_sorted    = self.scores_.argsort()[::-1][:self.k]
    self.columns_keep = [var_nam[i] for i in var_ndx_sorted]
    self.columns_drop = list(set(X.columns)-set(self.columns_keep))
    return self
  
  def transform(self, X):
    return X[self.columns_keep]

# COMMAND ----------

# MAGIC %md ## 3. Feature Reduction Model `PCA`

# COMMAND ----------

from sklearn.decomposition   import PCA
from sklearn.base              import BaseEstimator,TransformerMixin

class FeatureSelectionPCA(PCA):
    def __init__(self, n_components=10
                ):
        self.n_components= n_components
        super().__init__(n_components=self.n_components)

    def fit(self, X, y=None):
        return super().fit(X)
      
    def transform(self, X, y=None):
        return super().transform(X) 

# COMMAND ----------

# MAGIC %md __The End__