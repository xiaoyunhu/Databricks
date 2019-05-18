# Databricks notebook source
# MAGIC %md This notebook contains feature creation classes. 

# COMMAND ----------

import pandas as pd

# COMMAND ----------

from sklearn.base import BaseEstimator,TransformerMixin
class CreateTargetVarDF(BaseEstimator,TransformerMixin):

  def __init__(self, var):
      self.var=var

  def transform(self,X):
      return pd.DataFrame({'target': X[self.var]})
    
  def fit(self, X, y=None):
    return self

# COMMAND ----------

from sklearn.base import BaseEstimator,TransformerMixin
class CreateDatetimeVarsDF(BaseEstimator,TransformerMixin):
  def __init__(self, var='datetime', var_list=None):
    self.datetime_var = var
    if var_list is None:
      self.var_list=['year','month','day','hour','weekofyear','weekday','dayofyear']
    else:
      self.var_list=var_list

  def transform(self,X):
    return pd.DataFrame(data={'year'      : X[self.datetime_var].dt.year,
                              'month'     : X[self.datetime_var].dt.month,
                              'day'       : X[self.datetime_var].dt.day,
                              'hour'      : X[self.datetime_var].dt.hour,
                              'weekofyear': X[self.datetime_var].dt.weekofyear,
                              'weekday'   : X[self.datetime_var].dt.weekday,
                              'dayofyear' : X[self.datetime_var].dt.dayofyear}
                       )[self.var_list]
    
  def fit(self, X, y=None):
    return self

# COMMAND ----------

from sklearn.base import BaseEstimator,TransformerMixin
class CreateLagVarsDF (BaseEstimator,TransformerMixin):
  def __init__(self,var_list,lag_list=[1]):
    self.var_list=var_list
    self.lag_list=lag_list
     
  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    var_list_remaining = list(set(self.var_list).intersection(set(X.columns)))
    return pd.concat([pd.DataFrame(data={var+'_lag{lag}'.format(lag=lag): X[var].shift(lag)})
                      for var in var_list_remaining
                      for lag in self.lag_list
                     ],
                     axis=1)

# COMMAND ----------

from sklearn.base import BaseEstimator,TransformerMixin
class CreateRollingVarsDF (BaseEstimator,TransformerMixin):
  def __init__(self,var_list=None,lag_list=[1],win_list=[2]):
    self.var_list=var_list
    self.lag_list=lag_list
    self.win_list=win_list
     
  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    var_list_remaining = list(set(self.var_list).intersection(set(X.columns)))
    return pd.concat([pd.DataFrame(data={var+'_lag{lag}win{win}'.format(lag=lag,win=win): X[var].shift(lag).rolling(win).mean()})
                      for var in var_list_remaining
                      for lag in self.lag_list
                      for win in self.win_list
                     ],
                     axis=1)

# COMMAND ----------

from sklearn.base import BaseEstimator,TransformerMixin
class DropNaRowsDF (BaseEstimator,TransformerMixin):
  def __init__(self, how='any'):
    self.how=how

  def fit(self, X, y=None):
    return self
  
  def transform(self, X, y=None):
    return X.dropna(axis=0, 
                    how=self.how)

# COMMAND ----------

from sklearn.base import BaseEstimator,TransformerMixin
class FeatureUnionDF (BaseEstimator,TransformerMixin):
  def __init__(self, transformer_list):
    self.transformer_list=transformer_list
    
  def fit(self, X, y=None):
    return self 
  
  def transform(self, X, y=None):
    return pd.concat([transformer[1].fit(X).transform(X) 
                      for transformer 
                      in  self.transformer_list],
                     axis=1)

# COMMAND ----------

from sklearn.base import BaseEstimator,TransformerMixin
class CreateNamedVarsDF(BaseEstimator,TransformerMixin):

  def __init__(self, var_list=None, except_list=[]):
    self.var_list=var_list
    self.except_list=except_list
      
  def transform(self,X):
    return pd.DataFrame(X[self.var_list])
    
  def fit(self, X, y=None):
    if self.var_list is None:
      self.var_list = list(set(X.columns) - set(self.except_list))
    else:
      self.var_list = list(set(X.columns).intersection(set(self.var_list)) - set(self.except_list))
    return self

# COMMAND ----------

# MAGIC %md ## Not used

# COMMAND ----------

from sklearn.base import BaseEstimator,TransformerMixin
class FillNaColumnsDF(BaseEstimator,TransformerMixin):
  def __init__(self, method='pad', columns=None):
    self.method=method
    self.columns=columns
    
  def transform(self, X, y=None):
    if self.columns is None: 
      return X.fillna(method=self.method)
    else:
      from copy import deepcopy
      X_copy = deepcopy(X)
      X_copy[self.columns] = X_copy[self.columns].fillna(method=self.method)
      return X_copy
  
  def fit(self, X, y=None):
    return self

# COMMAND ----------

from sklearn.base import BaseEstimator,TransformerMixin
class DropNaColumnsDF(BaseEstimator,TransformerMixin):
  def __init__(self, how='all', limit=None):
    self.how  =how
    self.limit=limit
    
  def transform(self, X, y=None):
    return X.dropna(axis=1, 
                    how=self.how)
  
  def fit(self, X, y=None):
    return self