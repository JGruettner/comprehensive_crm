#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import pandas as pd
import sqlite3

import warnings
warnings.filterwarnings("ignore")

# Connect Data Sources
conn = sqlite3.connect('northwind.db')
orders = pd.read_sql_query("select * from Orders;", conn)
order_detail = pd.read_sql_query("select * from OrderDetails;", conn)
customer = pd.read_sql_query("select * from Customers;", conn)

order_detail.head()


# In[2]:


# Joining Tables to MasterDataSet
mds = pd.merge(orders, order_detail, on='OrderID', how='left')
mds2 = pd.merge(mds, customer, on='CustomerID', how='left')
mds2.info()


# In[3]:


# Amount of Orders per year
mds2['OrderDate'] = pd.to_datetime(mds2['OrderDate'])
mds2['Orderyear'] = mds2['OrderDate'].dt.year
print(mds2['Orderyear'].value_counts()) 


# In[4]:


#Calulate total purchase (aga revenue)
mds2['TotalPurchase'] = (mds2['Quantity'] * mds2['UnitPrice']) - (mds2['Quantity'] * mds2['UnitPrice']* mds2['Discount'])
mds2[['TotalPurchase', 'Quantity', 'UnitPrice', 'Discount']].head()


# In[5]:


import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

forecast = mds2[['OrderDate', 'TotalPurchase']]
forecast.index = pd.DatetimeIndex(pd.to_datetime(forecast['OrderDate']))
forecast = forecast.sort_index(ascending = False)

plot_data = [
    go.Scatter(
        x=mds2['OrderDate'],
        y=mds2['TotalPurchase'],
    )
]
plot_layout = go.Layout(
        title='Total Purchase'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[6]:


# We want to predict revenue for the next weeks
forecast2=forecast.resample('W').sum()
forecast2.tail()


# In[7]:


# according to prophet documentation every variables should have specific names
forecast2['ds'] = pd.to_datetime(forecast2.index)
forecast2 = forecast2.rename(columns = {'TotalPurchase': 'y'})
forecast2.head()


# In[8]:


from fbprophet import Prophet

# set the uncertainty interval to 95% (the Prophet default is 80%)
model = Prophet(yearly_seasonality = True, weekly_seasonality = False, interval_width = 0.85)
model.fit(forecast2)
sales_pred = model.predict(forecast2)


# In[10]:


# Merge predicted and original value
sales= pd.merge(forecast2, sales_pred, how = 'inner', on = 'ds')
sales[['y', 'yhat']].astype(int).head(10)


# In[11]:


#Calculate Accuracy Score
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(sales['y'], sales['yhat']))
print("Standardabweichung: ",sales['y'].std())


# In[12]:


import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_true=sales['y'], y_pred=sales['yhat'])


# In[13]:


# dataframe that extends into future 8 weeks 
future_dates = model.make_future_dataframe(freq = "w", periods = 8, include_history=True)

print("Prediction Times")
future_dates.tail()


# In[14]:


# predictions
sales_future = model.predict(future_dates)
model.plot(sales_future)


# In[15]:


#One other particularly strong feature of Prophet is its ability to return the components of our forecasts.
model.plot_components(sales_future)


# In[16]:


import pickle 
pickle.dump(model, open('model.pkl','wb'))


# In[ ]:


# Create API using flask
from flask import Flask, request, jsonify
app = Flask(__name__)

# Load the model from pickle
model_pkl = pickle.load(open('model.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using json request data
    prediction = model.predict(([[np.array(data['ds'])]]))

    # Take the first value of prediction
    output = prediction[0]

    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=False)


# In[ ]:


#Send an example to the REST API
import requests
import json
url = 'http://127.0.0.1:5000/api'

payload = {'ds': '2018-06-03'}
headers = {'content-type': 'application/json'}

#r = requests.post(url, data=json.payload, headers=headers)


# Jsonify the dataset
#r = requests.post(url,json={np.array(sales_future.values)})
#print(r.json())


# In[ ]:





# In[ ]:




