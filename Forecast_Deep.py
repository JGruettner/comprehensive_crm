#!/usr/bin/env python
# coding: utf-8

# # Customer lifetime value (CLV) can help to answer the most important questions about sales:
# 
# - How to Identify the most profitable customers?
# - How can a company offer the best product to the right customer?
# - How to approach profitable customers?
# - Determine the characteristics of most valuable customer relationships and seek out customers with similar traits
# - Predict future purchases and profits per customer
# - Deliver unique segment-specific marketing Treatments
# - Forecast customer satisfaction
# - Innovate and optimize marketing tools, tactics and channels
# - Adjust communication campaigns and messages
# - Cross-sell and up-sell based on individual patterns of buying

# In[ ]:


from datetime import date
import pandas as pd
import sqlite3
import lifetimes
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Connect Data Sources
conn = sqlite3.connect('northwind.db')
orders = pd.read_sql_query("select * from Orders;", conn)
order_detail = pd.read_sql_query("select * from OrderDetails;", conn)
customer = pd.read_sql_query("select * from Customers;", conn)

order_detail.head()


# In[ ]:


# Joining Tables to MasterDataSet
mds = pd.merge(orders, order_detail, on='OrderID', how='left')
mds2 = pd.merge(mds, customer, on='CustomerID', how='left')
mds2.info()


# In[ ]:


#Calculate total purchase aga revenue
mds2['TotalPurchase'] = (mds2['Quantity'] * mds2['UnitPrice']) - (mds2['Quantity'] * mds2['UnitPrice']* mds2['Discount'])
mds2['OrderDate'] = pd.to_datetime(mds2['OrderDate'])
mds2['CustomerID'].shape


# In[ ]:


# Calculate basic RFM Score manually
#T = Zeit zwischen der ersten Bestellung eines Kunden und dem Ende des Merkmalzeitraums

customer_detail=mds2.groupby('CustomerID').agg({'OrderDate': lambda date: (date.max() - date.min()).days,
                                        'OrderID': lambda num: len(np.unique(num)),
                                        'Quantity': lambda quant: quant.sum(),
                                        'TotalPurchase': lambda price: price.sum()})
customer_detail.columns=['recency','frequency','num_units','monetary']
customer_detail['frequency'] = customer_detail['frequency'].apply(lambda x: x-1) # Nummer minus 1
customer_detail['T'] = (mds2['OrderDate'].max()- mds2.groupby('CustomerID')['OrderDate'].min()).dt.days
customer_detail['avg_order_value']=customer_detail['monetary']/customer_detail['frequency']
customer_detail.head(20)


# In[ ]:


# Zu Sicherheit die Programmierung des Backups
from lifetimes.utils import summary_data_from_transaction_data
backup = summary_data_from_transaction_data(mds2, 'CustomerID', 'OrderDate', observation_period_end='2018-05-06')

print(backup.head())


# In[ ]:


# Summarise into RFM-Score
customer_detail['r_quartile'] = pd.qcut(customer_detail['recency'], 4, ['1','2','3','4'])
customer_detail['f_quartile'] = pd.qcut(customer_detail['frequency'], 4, ['4','3','2','1'])
customer_detail['m_quartile'] = pd.qcut(customer_detail['monetary'], 4, ['4','3','2','1'])
customer_detail['RFM_Score'] = customer_detail.r_quartile.astype(str)+ customer_detail.f_quartile.astype(str) + customer_detail.m_quartile.astype(str)
customer_detail['profit']=customer_detail['monetary']*0.05
customer_detail= customer_detail.query('frequency > 0')
customer_detail['Total_Score'] = customer_detail.r_quartile.astype(int)+ customer_detail.f_quartile.astype(int) + customer_detail.m_quartile.astype(int)

CountStatus = pd.value_counts(customer_detail['Total_Score'].values, sort=False)
CountStatus.plot.barh()


# # Get into Customer Lifetime Value

# - The Customer Lifetime Value refers to past profits + expected no. of future transactions * expected profit per transaction
# 
# **Estimate future purchase frequency per customer**
# - or in other words: the probability that a customer places a repeat purchase
# - Pareto/NBD for estimating the number of future purchases a customer will make
# - Beta Geometric/Negative Binomial Distribution (BG/NBD) model is an improvement of the Pareto/NBD model 
# 
# **Afterwards we estimate a customer’s average order value, to get the monetary involved**
# - a Gamma Gamma model to estimate average order value 
# - Gamma-Gamma submodel is used on top of the BG/NBD model to estimate the monetary value of transactions
# - We can only rely on a customer’s past purchases and characterizing events, like website visits, reviews, etc.
# 
# - plot_probability_alive_matrix

# In[ ]:


#https://lifetimes.readthedocs.io/en/master/lifetimes.html
from lifetimes import ModifiedBetaGeoFitter

mbgnbd = ModifiedBetaGeoFitter(penalizer_coef=0.001)
mbgnbd.fit(customer_detail['frequency'], customer_detail['recency'], customer_detail['T'], verbose=True, fit_method='Nelder-Mead')
mbgnbd.summary


# In[ ]:


from lifetimes import BetaGeoFitter
bgf = BetaGeoFitter(penalizer_coef=0.0001)
bgf.fit(customer_detail['frequency'], customer_detail['recency'], customer_detail['T'])
bgf.summary


# In[ ]:


from lifetimes.plotting import plot_frequency_recency_matrix
plot_frequency_recency_matrix(mbgnbd)


# In[ ]:


from lifetimes.plotting import plot_probability_alive_matrix
plot_probability_alive_matrix(bgf)


# In[ ]:


from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)


# In[ ]:


t = 90 # days to predict in the future 
customer_detail['pred_90d_bgf'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, 
                                                                                      customer_detail['frequency'], 
                                                                                      customer_detail['recency'], 
                                                                                      customer_detail['T'])
customer_detail.sort_values(by='pred_90d_bgf').tail(5)


# In[ ]:


# Get expected and actual repeated cumulative transactions.
from lifetimes.utils import expected_cumulative_transactions


# In[ ]:


#highest expected purchases in the next period

customer_detail['pred_90d_mbgnbd'] = mbgnbd.conditional_expected_number_of_purchases_up_to_time(t, 
                                                                                      customer_detail['frequency'], 
                                                                                      customer_detail['recency'], 
                                                                                      customer_detail['T'])
customer_detail.head()


# In[ ]:


customer_detail['p_alive'] = mbgnbd.conditional_probability_alive(customer_detail['frequency'], customer_detail['recency'], customer_detail['T'])
customer_detail.head()


# In[ ]:


#The Gamma-Gamma modelassumes that there is no relationship between the monetary value and the purchase frequency
customer_detail[['avg_order_value','frequency']].corr()


# In[ ]:


#It is used to estimate the average monetary value of customer transactions
from lifetimes import GammaGammaFitter

gg = GammaGammaFitter(penalizer_coef = 0.001)
gg.fit(customer_detail['frequency'], customer_detail['avg_order_value'],verbose=True)

print(gg.conditional_expected_average_profit(
        customer_detail['frequency'],
        customer_detail['avg_order_value']
    ).head(10))


# In[ ]:


customer_detail['clv']=gg.customer_lifetime_value(
    mbgnbd,
    customer_detail['frequency'],
    customer_detail['recency'],
    customer_detail['T'],
    customer_detail['avg_order_value'],
    time=t,
    discount_rate=0
).astype(int)
customer_detail.head()


print(gg.conditional_expected_average_profit(
        customer_detail['frequency'],
        customer_detail['monetary']
    ).head(10))


# In[ ]:




