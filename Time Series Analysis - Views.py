#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
from datetime import datetime
rcParams['figure.figsize'] = 10, 6


# In[2]:


dataset = pd.read_csv('Views.csv', encoding='utf-16', sep = '\t')
dataset = dataset.iloc[:, 2:4]
dataset['Date'] = pd.to_datetime(dataset['Date'], infer_datetime_format = True)
indexedDataset = dataset.set_index(['Date'])


# In[3]:


#dataset['Date']
indexedDataset.tail()


# In[4]:


plt.xlabel('Date')
plt.ylabel('Total Views')
plt.plot(indexedDataset)


# In[16]:


# Find the rolling statistics
rollingMean = indexedDataset.rolling(window = 12).mean()

rollingSTD = indexedDataset.rolling(window = 12).std()
print(rollingMean, rollingSTD)


# In[17]:


# Plot Rolling Stats

orig = plt.plot(indexedDataset, color = 'blue', label='Original')
mean = plt.plot(rollingMean, color = 'green', label='Rolling Mean')
std = plt.plot(rollingSTD, color='red', label='Rolling STD')
plt.title('Rolling Stats')
plt.show(block=False)


# In[18]:


# Perform Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller

print('Results of Dickey-Fuller Test:')
dftest = adfuller(indexedDataset['# Analytics Viewed'], autolag = 'AIC')

dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
    
print(dfoutput)


# In[19]:


# Estimate the trend
indexedDataset_logScale = np.log10(indexedDataset)
plt.plot(indexedDataset_logScale)


# In[20]:


movingAverage = indexedDataset_logScale.rolling(window = 12).mean()
movingSTD = indexedDataset_logScale.rolling(window = 12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage, color='red')
plt.plot(movingSTD, color = 'black')


# In[21]:


def test_stationary(timeseries):
    movingAverage = timeseries.rolling(window = 12).mean()
    movingSTD = timeseries.rolling(window = 12).std()
    
    #plot rolling stats:
    
    orig = plt.plot(timeseries, color = 'blue', label='Original')
    mean = plt.plot(movingAverage, color = 'green', label='Rolling Mean')
    std = plt.plot(movingSTD, color='red', label='Rolling STD')
    plt.title('Rolling Stats')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries['# Analytics Viewed'], autolag = 'AIC')

    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)


# In[22]:


datasetLDShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(datasetLDShifting)


# In[23]:


datasetLDShifting.dropna(inplace = True)
test_stationary(datasetLDShifting)


# In[24]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_logScale)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale, label = 'Original')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationary(decomposedLogData)


# In[39]:


from statsmodels.tsa.arima_model import ARIMA

#AR Model
model = ARIMA(indexedDataset_logScale, order=(2,1,2))
results_AR = model.fit(disp = -1)
plt.plot(datasetLDShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_AR.fittedvalues-datasetLDShifting['# Analytics Viewed'])**2))
print('Plotting AR Model')


# In[40]:


#MA Model
model = ARIMA(indexedDataset_logScale, order=(2,1,2))
results_MA = model.fit(disp = -1)
plt.plot(datasetLDShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_MA.fittedvalues-datasetLDShifting['# Analytics Viewed'])**2))
print('Plotting MA Model')


# In[41]:


model = ARIMA(indexedDataset_logScale, order=(2,1,2))
results_ARIMA = model.fit(disp = -1)
plt.plot(datasetLDShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues-datasetLDShifting['# Analytics Viewed'])**2))
print('Plotting ARIMA Model')


# In[42]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())


# In[43]:


#Convert to cummulative sum
preductions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(preductions_ARIMA_diff_cumsum.head())


# In[44]:


predictions_ARIMA_log = pd.Series(indexedDataset_logScale['# Analytics Viewed'].ix[0], index=indexedDataset_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(preductions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()


# In[45]:


predictions_ARIMA= np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)


# In[46]:


#indexedDataset_logScale.head() - 113rows
results_ARIMA.plot_predict(1, 125)
x=results_ARIMA.forecast(steps=12)


# In[47]:


results_ARIMA.forecast(steps = 120)


# In[48]:


results = results_ARIMA.forecast(steps = 12)
res=[]
for r in results:
    res.append(10**r)


# In[49]:


print(res[0])

