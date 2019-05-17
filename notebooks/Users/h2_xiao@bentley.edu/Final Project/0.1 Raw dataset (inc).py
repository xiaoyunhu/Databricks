# Databricks notebook source
# MAGIC %md ## 1.1 BCI dataset

# COMMAND ----------

# MAGIC %python
# MAGIC import pandas as pd
# MAGIC bci_pdf = pd.read_csv('/dbfs/mnt/group-ma707/data/5tc_plus_ind_vars.csv') \
# MAGIC             .rename(columns={'P3A~IV':'P3A_IV'}) \
# MAGIC             .assign(date=lambda pdf: pd.to_datetime(pdf.Date)) \
# MAGIC             .drop('Date', axis=1) \
# MAGIC             .sort_index(ascending=True)
# MAGIC bci_pdf.columns = bci_pdf.columns.str.lower()
# MAGIC bci_pdf.info()

# COMMAND ----------

# MAGIC %md ## 1.2 Coal dataset

# COMMAND ----------

# MAGIC %python
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC coal_pdf = \
# MAGIC pd.read_csv('/dbfs/mnt/group-ma707/data/mining_com_coal.csv', 
# MAGIC             encoding='ISO-8859-1'
# MAGIC            ) \
# MAGIC   .dropna(axis=0, subset=['tags','content','title']) \
# MAGIC   .assign(date   =lambda pdf: pd.to_datetime(pd.to_datetime(pdf.date).dt.date)
# MAGIC          ) \
# MAGIC   .loc[:,['date','tags','title','content']] \
# MAGIC   .sort_values('date', ascending=True)
# MAGIC coal_pdf.info()

# COMMAND ----------

# MAGIC %python 
# MAGIC dup_date_ser = \
# MAGIC coal_pdf['date'] \
# MAGIC   .value_counts() \
# MAGIC   .sort_index(ascending=True) \
# MAGIC   .loc[lambda ser:ser==2] 
# MAGIC dup_date_ser \
# MAGIC   .tail()

# COMMAND ----------

dup_date = dup_date_ser.index.date[-1]
dup_date

# COMMAND ----------

# MAGIC %python coal_pdf.set_index('date').loc[dup_date].tags

# COMMAND ----------

# MAGIC %python 
# MAGIC coal_pdf.groupby(by='date').sum().loc[dup_date].tags

# COMMAND ----------

# MAGIC %md Problem above. Fix below.
# MAGIC 
# MAGIC Put space at the end of each value in the `tags`, `title` and `content` columns. See code cell below.

# COMMAND ----------

# MAGIC %python
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC coal_pdf = \
# MAGIC pd.read_csv('/dbfs/mnt/group-ma707/data/mining_com_coal.csv', 
# MAGIC             encoding='ISO-8859-1'
# MAGIC            ) \
# MAGIC   .dropna(axis=0, subset=['tags','content','title']) \
# MAGIC   .assign(date   =lambda pdf: pd.to_datetime(pd.to_datetime(pdf.date).dt.date),
# MAGIC           tags   =lambda pdf: pd.Series(data=[tags   +' ' for tags    in list(pdf.tags)]),
# MAGIC           content=lambda pdf: pd.Series(data=[content+' ' for content in list(pdf.content)]),
# MAGIC           title  =lambda pdf: pd.Series(data=[title  +' ' for title   in list(pdf.title)])
# MAGIC          ) \
# MAGIC   .loc[:,['date','tags','title','content']] \
# MAGIC   .sort_values('date', ascending=True)
# MAGIC coal_pdf.info()

# COMMAND ----------

# MAGIC %python coal_pdf.set_index('date').loc[dup_date].tags

# COMMAND ----------

# MAGIC %python 
# MAGIC coal_pdf.groupby(by='date').sum().loc[dup_date].tags

# COMMAND ----------

# MAGIC %python
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC coal_pdf = \
# MAGIC pd.read_csv('/dbfs/mnt/group-ma707/data/mining_com_coal.csv', 
# MAGIC             encoding='ISO-8859-1'
# MAGIC            ) \
# MAGIC   .dropna(axis=0, subset=['tags','content','title']) \
# MAGIC   .assign(date   =lambda pdf: pd.to_datetime(pd.to_datetime(pdf.date).dt.date),
# MAGIC           tags   =lambda pdf: pd.Series(data=[tags   +' ' for tags    in list(pdf.tags)]),
# MAGIC           content=lambda pdf: pd.Series(data=[content+' ' for content in list(pdf.content)]),
# MAGIC           title  =lambda pdf: pd.Series(data=[title  +' ' for title   in list(pdf.title)])
# MAGIC          ) \
# MAGIC   .loc[:,['date','tags','title','content']] \
# MAGIC   .groupby(by='date') \
# MAGIC   .sum() \
# MAGIC   .reset_index() \
# MAGIC   .sort_values('date', ascending=True)
# MAGIC coal_pdf.info()

# COMMAND ----------

# MAGIC %python 
# MAGIC coal_pdf['date'] \
# MAGIC   .value_counts() \
# MAGIC   .sort_index(ascending=True) \
# MAGIC   .value_counts()

# COMMAND ----------

coal_pdf.date.dt.weekday.value_counts()

# COMMAND ----------

# MAGIC %python 
# MAGIC coal_pdf.set_index('date').sort_index(ascending=True).head(10) # debug

# COMMAND ----------

# MAGIC %python 
# MAGIC coal_pdf.set_index('date').sort_index(ascending=True).tags[-1] #.tail(10) # debug

# COMMAND ----------

# MAGIC %python 
# MAGIC coal_pdf.set_index('date').resample('D').pad().sort_index(ascending=True).tail(20)

# COMMAND ----------

# MAGIC %python 
# MAGIC coal_pdf.set_index('date').resample('D').pad().reset_index().date.dt.weekday.value_counts()

# COMMAND ----------

# MAGIC %python
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC coal_pdf = \
# MAGIC pd.read_csv('/dbfs/mnt/group-ma707/data/mining_com_coal.csv', 
# MAGIC             encoding='ISO-8859-1'
# MAGIC            ) \
# MAGIC   .dropna(axis=0, subset=['tags','content','title']) \
# MAGIC   .assign(date   =lambda pdf: pd.to_datetime(pd.to_datetime(pdf.date).dt.date),
# MAGIC           tags   =lambda pdf: pd.Series(data=[tags   +' ' for tags    in list(pdf.tags)]),
# MAGIC           content=lambda pdf: pd.Series(data=[content+' ' for content in list(pdf.content)]),
# MAGIC           title  =lambda pdf: pd.Series(data=[title  +' ' for title   in list(pdf.title)])
# MAGIC          ) \
# MAGIC   .loc[:,['date','tags','title','content']] \
# MAGIC   .groupby(by='date') \
# MAGIC   .sum() \
# MAGIC   .resample('D') \
# MAGIC   .pad() \
# MAGIC   .reset_index() \
# MAGIC   .sort_values('date', ascending=True) \
# MAGIC   .add_suffix('_coal') \
# MAGIC   .rename(columns={"date_coal": "date"})
# MAGIC coal_pdf.info()

# COMMAND ----------

# MAGIC %md ## 1.3 Iron ore dataset

# COMMAND ----------

# MAGIC %python
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC ore_pdf = \
# MAGIC pd.read_csv('/dbfs/mnt/group-ma707/data/mining_com_iron_ore.csv', 
# MAGIC             encoding='ISO-8859-1'
# MAGIC            ) \
# MAGIC   .loc[:,['date','tags','title','content']] \
# MAGIC   .fillna({'tags'   :'',
# MAGIC            'content':'',
# MAGIC            'title'  :''
# MAGIC           }) \
# MAGIC   .assign(date = lambda pdf: pd.to_datetime(pd.to_datetime(pdf.date,utc=True).dt.normalize().dt.date)) \
# MAGIC   .groupby(by='date') \
# MAGIC   .agg({'tags'   : lambda ser: ' '.join(ser),
# MAGIC         'content': lambda ser: ' '.join(ser),
# MAGIC         'title'  : lambda ser: ' '.join(ser)}) \
# MAGIC   .resample('D') \
# MAGIC   .pad() \
# MAGIC   .reset_index() \
# MAGIC   .add_suffix('_ore') \
# MAGIC   .rename(columns={"date_ore": "date"})
# MAGIC ore_pdf.info(10)

# COMMAND ----------

coal_pdf.info()

# COMMAND ----------

bci_pdf.info()

# COMMAND ----------

ore_pdf.info()

# COMMAND ----------

# MAGIC %md ## 1.4 Initial dataset (TBD)
# MAGIC 
# MAGIC The "initial dataset" is created by merging the BCI and the two mining datasets.
# MAGIC Below the coal and bci datasets are merged to create the `bci_coal_pdf` datasets/dataframe.

# COMMAND ----------

# MAGIC %python
# MAGIC import pandas as pd
# MAGIC bci_coal_pdf = \
# MAGIC pd.concat(objs=[ bci_pdf.set_index('date'), 
# MAGIC                 coal_pdf.set_index('date')], 
# MAGIC           join='inner',
# MAGIC           axis=1
# MAGIC          ) \
# MAGIC   .reset_index()
# MAGIC bci_coal_pdf.info()

# COMMAND ----------

# MAGIC %md ## 1.5 Initial dataset (TBD)
# MAGIC 
# MAGIC The "initial dataset" is created by merging the BCI and the two mining datasets.
# MAGIC Below the ironore and bci datasets are merged to create the `bci_ironore_pdf` datasets/dataframe.

# COMMAND ----------

# MAGIC %python
# MAGIC import pandas as pd
# MAGIC bci_ironore_pdf = \
# MAGIC pd.concat(objs=[bci_pdf.set_index('date'), 
# MAGIC                 ore_pdf.set_index('date')], 
# MAGIC           join='inner',
# MAGIC           axis=1
# MAGIC          ) \
# MAGIC   .reset_index()
# MAGIC bci_ironore_pdf.info()

# COMMAND ----------

# MAGIC %md ## 1.6 Initial dataset (TBD)
# MAGIC 
# MAGIC The "initial dataset" is created by merging the BCI and the two mining datasets.
# MAGIC Below the bci_pdf, coal_pdf and ore_pdf datasets are merged to create the `bci_all_dual_pdf` datasets/dataframe.

# COMMAND ----------

bci_dual_pdf = \
pd.concat(objs=[ bci_pdf.set_index('date'), 
                 ore_pdf.set_index('date'),
                 coal_pdf.set_index('date')], 
          join='inner',
          axis=1
         ) \
  .reset_index()
bci_dual_pdf.info()