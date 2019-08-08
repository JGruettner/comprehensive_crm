#!/usr/bin/env python
# coding: utf-8

# **goal for this script is to calculate fundamental KPI scores and to deploy a neuronal network predicting revenue**
# - Does the script calculate fundamental KPI scores for customers and products?
# - Does the revenue prediction use a neuronal network?

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

# In[1]:


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
products = pd.read_sql_query("select * from Products;", conn)
categories = pd.read_sql_query("select * from Categories;", conn)

order_detail.head()


# In[2]:


# Joining Tables to MasterDataSet
mds = pd.merge(orders, order_detail, on='OrderID', how='left')
mds2 = pd.merge(mds, customer, on='CustomerID', how='left')
mds2.info()


# In[3]:


#Calculate total purchase aga revenue
mds2['TotalPurchase'] = (mds2['Quantity'] * mds2['UnitPrice']) - (mds2['Quantity'] * mds2['UnitPrice']* mds2['Discount'])
mds2['OrderDate'] = pd.to_datetime(mds2['OrderDate'])
mds2.groupby('CustomerID').mean()


# In[4]:


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


# In[5]:


# Zu Sicherheit die Programmierung des Backups
from lifetimes.utils import summary_data_from_transaction_data
backup = summary_data_from_transaction_data(mds2, 'CustomerID', 'OrderDate', observation_period_end='2018-05-06')

print(backup.head())


# In[6]:


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
# - Gamma Gamma model estimates average order values 
# - this model is used on top of the BG/NBD model to estimate the monetary value of transactions
# - We can only rely on a customer’s past purchases and characterizing events, like website visits, reviews, etc.

# In[7]:


#https://lifetimes.readthedocs.io/en/master/lifetimes.html
from lifetimes import ModifiedBetaGeoFitter

mbgnbd = ModifiedBetaGeoFitter(penalizer_coef=0.001)
mbgnbd.fit(customer_detail['frequency'], customer_detail['recency'], customer_detail['T'], verbose=True, fit_method='Nelder-Mead')
mbgnbd.summary


# In[8]:


from lifetimes import BetaGeoFitter
bgf = BetaGeoFitter(penalizer_coef=0.0001)
bgf.fit(customer_detail['frequency'], customer_detail['recency'], customer_detail['T'])
bgf.summary


# In[9]:


#from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes.plotting import plot_frequency_recency_matrix
plot_frequency_recency_matrix(mbgnbd)


# In[10]:


from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)


# In[11]:


t = 90 # days to predict in the future 
customer_detail['pred_90d_bgf'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, 
                                                                                      customer_detail['frequency'], 
                                                                                      customer_detail['recency'], 
                                                                                      customer_detail['T'])
customer_detail.sort_values(by='pred_90d_bgf').tail(5)


# In[12]:


#highest expected purchases in the next period

customer_detail['pred_90d_mbgnbd'] = mbgnbd.conditional_expected_number_of_purchases_up_to_time(t, 
                                                                                      customer_detail['frequency'], 
                                                                                      customer_detail['recency'], 
                                                                                      customer_detail['T'])
customer_detail.head()


# In[13]:


customer_detail['p_alive'] = mbgnbd.conditional_probability_alive(customer_detail['frequency'], customer_detail['recency'], customer_detail['T'])
customer_detail.head()


# In[14]:


#The Gamma-Gamma model assumes that there is no relationship between the monetary value and the purchase frequency
customer_detail[['avg_order_value','frequency']].corr()


# In[15]:


#It is used to estimate the average monetary value of customer transactions
from lifetimes import GammaGammaFitter

gg = GammaGammaFitter(penalizer_coef = 0.001)
gg.fit(customer_detail['frequency'], customer_detail['avg_order_value'],verbose=True)

print(gg.conditional_expected_average_profit(
        customer_detail['frequency'],
        customer_detail['avg_order_value']
    ).head(10))


# In[16]:


customer_detail['clv']=gg.customer_lifetime_value(
    mbgnbd,
    customer_detail['frequency'],
    customer_detail['recency'],
    customer_detail['T'],
    customer_detail['avg_order_value'],
    time=t,
    discount_rate=0
).astype(int)
customer_detail[['frequency', 'pred_90d_bgf', 'monetary', 'avg_order_value', 'clv']].head()


# In[17]:


customer_detail['exp_orders'] = (customer_detail['clv']/gg.conditional_expected_average_profit(customer_detail['frequency'], customer_detail['avg_order_value'])).astype(int)
customer_detail['potential']=100-((100/customer_detail['clv'])*customer_detail['monetary'])
customer_detail[['frequency', 'exp_orders', 'monetary', 'avg_order_value', 'clv', 'potential']].head(10)


# In[18]:


#Da es sich um ein komplexes Modell handelt, ist der Zusammenhang nicht linear
plt.scatter(customer_detail['exp_orders'], customer_detail['clv'], alpha=0.5)
plt.show()


# In[19]:


# Currently the dataset provides 47 attributes to predict the revenue
mds3 = pd.merge(mds2, customer_detail, on='CustomerID', how='left')
#mds3.shape
mds3.info()


# # Calculating basic product sales KPIs
# - Share of product category in sales
# - product category growth (Compound Annual Growth Rates, also known as CAGR)
# - price development (Exponential Moving Average (EMA),more weight to the recent prices)
# - instead of categories we could use products, but categories give a better overview

# In[20]:


prod_details=pd.merge(mds3[['OrderDate', 'ProductID', 'Quantity', 'TotalPurchase']], products, on='ProductID', how='left')
prod_details2 = pd.merge(prod_details, categories, on='CategoryID', how='left')
prod_details2['OrderYear'] = prod_details2['OrderDate'].dt.year
prod_details2.head()


# In[22]:


cat_details = prod_details2.groupby('CategoryName')['OrderYear'].value_counts().unstack()
print(prod_details2['OrderDate'].max())
d1 = date(2018, 1, 1)
cat_details[2018] = cat_details[2018].apply(lambda x: x*1.58333)
print(cat_details)


# In[23]:


cat_details2 =cat_details.pct_change(axis='columns')
cat_details2


# In[35]:


# bei gelegenheit über eine Funktion
cat_details2['growth'] =cat_details2[2017]*0.2+cat_details2[2018]*0.8
cat_details2


# In[33]:


cat_details.ewm(com=0.5, axis='columns').mean()


# In[26]:


cat_details3=cat_details2.join(pd.DataFrame((100/prod_details2['TotalPurchase'].sum())*prod_details2.groupby('CategoryName')['TotalPurchase'].sum()))
cat_details3[mov_growth] = cat_details3
cat_details3 = cat_details.drop([2016], axis=1)


# In[ ]:


cat_details = pd.DataFrame((100/prod_details2['TotalPurchase'].sum())*prod_details2.groupby('CategoryName')['TotalPurchase'].sum())


# In[ ]:





# In[ ]:


print(prod_details2['OrderDate'].max())
d1 = date(2018, 1, 1)
cat_details2[2018] = cat_details2[2018].apply(lambda x: x*1.58333)
print(cat_details2)


# In[ ]:


cat_details2['CAGR'] = cat_details2.T.pct_change().add(1).prod().pow(1./(len(cat_details2.columns) - 1)).sub(1)
cat_details2


# In[ ]:


cat_details2.pct_change(axis='columns')


# In[ ]:


from scipy import stats
from sklearn.linear_model import LinearRegression
X = cat_details2[[2016,2017,2018]]
y = cat_details2['TotalPurchase']
#cat_details2['growth_rate'] = stats.linregress(X,y)
#cat_details2['growth_rate']=np.polyfit(X,y)
lm = LinearRegression()
model = lm.fit(X, y)
lm.get_params
#model.coef_


# In[ ]:




