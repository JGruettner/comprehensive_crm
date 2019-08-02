#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import pandas as pd
import sqlite3
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Connect Datasources
conn = sqlite3.connect('northwind.db')
orders = pd.read_sql_query("select * from Orders;", conn)
order_detail = pd.read_sql_query("select * from OrderDetails;", conn)
customer = pd.read_sql_query("select * from Customers;", conn)

order_detail.head()


# In[2]:


# Merge to MasterDataSet
mds = pd.merge(orders, order_detail, on='OrderID', how='left')
mds2 = pd.merge(mds, customer, on='CustomerID', how='left')
mds2.info()


# In[3]:


#Calculate total purchase aga revenue
mds2['TotalPurchase'] = (mds2['Quantity'] * mds2['UnitPrice']) - (mds2['Quantity'] * mds2['UnitPrice']* mds2['Discount'])
mds2[['TotalPurchase', 'Quantity', 'UnitPrice', 'Discount']].head()


# In[4]:


# We assume that distribution success to different countries have a major influence on revenue
import seaborn as sns
sns.set(style="whitegrid", palette="muted")
# Draw a categorical scatterplot to show each observation
sns.swarmplot(x="OrderDate", y="TotalPurchase", hue="ShipCountry", data=mds2)


# In[5]:


mds2['OrderDate'] = pd.to_datetime(mds2['OrderDate'])
mds2['Orderyear'] = mds2['OrderDate'].dt.year
mds2.groupby('Orderyear')['ShipCountry'].value_counts().unstack()


# In[6]:


# Umwandlung eines kategorialen Merkmals in ein binäres
countries = pd.get_dummies(mds2['ShipCountry'])
mds2 = mds2.drop('ShipCountry',axis = 1)
mds3 = mds2.join(countries)

mds3.head()


# In[7]:


# Again - we would like to forecast weeks
mds3.index= pd.DatetimeIndex(mds3['OrderDate']) 

weekly = pd.DataFrame()
weekly['revenue'] = mds3.TotalPurchase.resample('W').sum()
weekly['avg_quantity'] = mds3.Quantity.resample('W').mean()
weekly['sum_orders'] = mds3.OrderID.resample('W').count()
weekly['avg_unitprice'] = mds3.UnitPrice.resample('W').mean()
weekly['USA'] = mds3.USA.resample('W').sum()
weekly['Germany'] = mds3.Germany.resample('W').sum()
weekly['Brazil'] = mds3.Brazil.resample('W').sum()
weekly['date'] = pd.to_datetime(weekly.index)


weekly = weekly.sort_index(ascending = False)
print(weekly.tail(5))
print(weekly['avg_quantity'].plot())


# ## Stationarity is required for VAR

# In[8]:


#The ADF number should be a negative number
#less than 0.5 and can assume our data is stationary.
#p-value > 0.05: the data is non-stationary

from arch.unitroot import ADF
adf_revenue = ADF(weekly['revenue'])
adf_avg_quantity = ADF(weekly['avg_quantity'])
adf_sum_orders = ADF(weekly['sum_orders'])
adf_avg_unitprice = ADF(weekly['avg_unitprice'])
adf_USA = ADF(weekly['USA'])
adf_Germany = ADF(weekly['Germany'])
adf_Brazil = ADF(weekly['Brazil'])
print('revenue: {0:0.4f}'.format(adf_revenue.pvalue))
print('avg_quantity: {0:0.4f}'.format(adf_avg_quantity.pvalue))
print('sum_orders: {0:0.4f}'.format(adf_sum_orders.pvalue))
print('avg_unitprice: {0:0.4f}'.format(adf_avg_unitprice.pvalue))
print('USA: {0:0.4f}'.format(adf_Germany.pvalue))
print('Germany: {0:0.4f}'.format(adf_revenue.pvalue))
print('Brazil: {0:0.4f}'.format(adf_Brazil.pvalue))
#print(adf.summary().as_text())


# In[9]:


transformed = weekly.diff()
transformed.head(2)


# In[10]:


# Calculate diff the "long way"
weekly['lag-1'] = weekly['revenue'].shift(1)
weekly['lag-1'] = weekly['lag-1'].fillna(weekly['revenue'])
weekly['y'] = (weekly['revenue']-weekly['lag-1']).astype('int')

weekly.head()


# In[11]:


weekly['date'] = pd.to_datetime(weekly['date'])
import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
plot_data = [
    go.Scatter(
        x=weekly['date'],
        y=weekly['y'],
    )
]
plot_layout = go.Layout(
        title='Weekly Sales'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[12]:


#Quick look at the parameters
weekly1 = weekly.drop(['date', 'lag-1'], axis=1)

print(weekly1.shape)
print(pd.infer_freq(weekly1.index))


# # First: Vector Autoregression (VAR)

# In[13]:


#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR

var_model = VAR(endog=weekly1, dates=weekly1.index, freq=pd.infer_freq(weekly1.index))
# here the lag order is important
var_res = var_model.fit(maxlags=None, method='ols', ic=None, trend='c', verbose=True)
var_res.summary()


# In[14]:


# Calculate dates forecasting next 2 months
future_dates=pd.date_range(weekly1.index.max(), periods=8, freq='-1W-SUN')
future_dates


# In[15]:


yhat = var_res.forecast(var_res.y, steps=8)
weekly2 = pd.DataFrame(data=yhat, columns=weekly1.columns, index=future_dates)
weekly2.head(10)


# In[16]:


#var_res.plot_forecast(8)
weekly_pred=var_res.fittedvalues
weekly_pred=weekly_pred.add_prefix('pred_')
weekly_pred.head(5)


# In[17]:


weekly_comp=pd.merge(weekly1, weekly_pred, left_index=True, right_index=True)
weekly_comp[['y', 'pred_y']].astype(int).plot()


# In[18]:


import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_true=weekly_comp['avg_quantity'], y_pred=weekly_comp['pred_avg_quantity'])


# # Some Sleight (or Taschenspielertrick)

# In[19]:


# The problem with trees is that they cant predict on future attributes that doesn exist
weekly_pred2 = weekly_pred.drop(['pred_y'], axis=1)
weekly_pred2.head(10) #as test dataset


# In[24]:


weekly_orig =weekly.drop(['date', 'lag-1', 'y'], axis=1)
X_train = weekly_orig.loc[:, weekly_orig.columns != 'revenue']
y_train = weekly_orig.loc[:, weekly_orig.columns == 'revenue']
X_test = weekly_pred2.loc[:, weekly_pred2.columns != 'pred_revenue']
y_test = weekly_pred2.loc[:, weekly_pred2.columns == 'pred_revenue']
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[40]:


from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error  
from time import time  

t = time() 
rf=RandomForestRegressor(n_estimators=120, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False).fit(X_train,y_train) 
y_pred3 = rf.predict(X_test) 

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mean_absolute_percentage_error(y_true=y_train.loc[y_train.index=='2018-05-06', :], y_pred=y_pred3))
print('Benötigte Zeit: {} mins'.format(round((time() - t) / 60, 2)))  


# In[38]:


rf_coef = pd.DataFrame(data=np.round_(rf.feature_importances_, decimals=3), index=X_train.columns)   
rf_coef.sort_values(by=0, ascending=False)


# In[41]:


from sklearn.ensemble import GradientBoostingRegressor

t = time() 
gb = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=5, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001).fit(X_train,y_train)
y_pred4 = gb.predict(X_test) 

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mean_absolute_percentage_error(y_true=y_train.loc[y_train.index=='2018-05-06', :], y_pred=y_pred4))
print('Benötigte Zeit: {} mins'.format(round((time() - t) / 60, 2)))  


# # Better results with Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)

# In[ ]:


import statsmodels.api as sm
exog = weekly['y']
endog = weekly['avg_quantity', 'sum_orders','avg_unitprice', 'USA', 'Germany', 'Brazil']
varmax_model = sm.tsa.VARMAX(endog=endog, order=(2,0), trend='nc', exog=exog)
#res = mod.fit(maxiter=1000, disp=False)

#print(res.summary())


# In[ ]:



predicted_result = res.predict(start=0, end=110)
res.plot_diagnostics()


#https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_varmax.html

#predicted_result = res.predict(start=0, end=95)
#res.plot_diagnostics()


# Predicting closing price of Google and microsoft
#train_sample = weekly['diff'].values
#model = sm.tsa.VARMAX(train_sample,trend='c')
#äresult = model.fit(maxiter=1000,disp=False)
#print(result.summary())


# In[ ]:


#converting predictions to dataframe
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,13):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

#check rmse
for i in cols:
    print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[i], valid[i])))



