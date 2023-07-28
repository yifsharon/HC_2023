# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:18:29 2023

@author: YIFEI YAO
"""

# %%

# 压缩极值 WM
def compress_outliers_mad(data,n=3,m=5):
    '''压缩极值，将n倍mad以上的数据定义为极端值，将超过n倍mad的数，压缩到n-m倍mad之间，能保留数据之间的大小关系'''
    #-----------------------------------------------
    # data = [1, 2, 3, -100, 4, 5, 200, 6, 7,np.nan]
    # compressed_data = compress_outliers(data)
    # print(compressed_data)
    #-----------------------------------------------
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))
    threshold = n * 1.4826 * mad  # 超过3倍 MAD 的阈值
    compressed_data = data.copy()

    # 找到超过3倍 MAD 的索引
    outliers_index = np.where(np.abs(data - median) > threshold)[0]

    if len(outliers_index) > 0:
        max_outlier = np.take(compressed_data, outliers_index, mode='clip').max()
        min_outlier = np.take(compressed_data, outliers_index, mode='clip').min()
        compressed_data = np.where(compressed_data > median + threshold, median + threshold + (m-n) * 1.4826 * mad * (compressed_data - median - threshold) / (max_outlier - median - threshold),compressed_data)
        compressed_data = np.where(compressed_data < median - threshold, median - threshold - (m-n) * 1.4826 * mad * (compressed_data - median - threshold) / (min_outlier - median - threshold),compressed_data)
        compressed_data = np.where(np.abs(compressed_data-median)> np.abs(data-median),data,compressed_data)
    return compressed_data

# %%
#去极值
def winsorize(dfcol,method='boxplot',nsigma = 3,niqr=1.5):
    
    eligible_method = ['boxplot','medcouple','nsigma_mean','nsigma_median','quantile','median']
    if method not in eligible_method:
        raise AttributeError('method must be one of %s' % ','.join(eligible_method))
        return
           
    stats = dfcol.describe()
    mu,sig = stats.loc['mean'], stats.loc['std']
    q3,q2,q1 = stats.loc['75%'],stats.loc['50%'],stats.loc['25%']
    iqr = q3 - q1
    
    if method == 'nsigma_mean':            
        upp = mu + nsigma * sig
        low = mu - nsigma * sig
        
    elif method == 'boxplot':
        upp = q3 + niqr * iqr
        low = q1 - niqr * iqr
            
    elif method == 'medcouple':
        high_list = list(filter(lambda x:x>q2, list(dfcol)))
        low_list = list(filter(lambda x:x<q2, list(dfcol)))
        mc_list = [(x + y - 2*q2)/(x - y) for (x,y) in zip(high_list,low_list)]
        mc = np.median(mc_list)
        low = q1 - 1.5 * np.exp((-3.5 - 0.5 *(mc<0)) *mc) * iqr
        upp = q3 + 1.5 * np.exp((3.5 + 0.5 *(mc>=0)) *mc) * iqr            
    
    elif method == 'nsigma_median':
        upp = q2 + nsigma * sig
        low = q2 - nsigma * sig
   
    elif method == 'quantile':
        gap_upper,gap_lower = q3-q2,q2-q1
        upp = q3 + 1.5 * gap_upper
        low = q1 - 1.5 * gap_lower

    elif method == 'median':
        abs_list = [abs(x - q2) for x in list(dfcol)]
        med = np.median(abs_list)
        upp = q2 + 5.2 * med
        low = q2 - 5.2 * med
        
    res = np.clip(dfcol,low,upp)
    return res, upp, low


# %%

#标准化
def standardize(dfcol,method='zscore'):
    
    eligible_method = ['zscore','range_standard','range_regular','div_max','min_div','weight','sigmoid','percentile']
    if method not in eligible_method:
        raise AttributeError('method must be one of %s' % ','.join(eligible_method))
        return
           
    stats = dfcol.describe()
    mu,sig = stats.loc['mean'], stats.loc['std']
    M,m = stats.loc['max'], stats.loc['min']
    n = stats.loc['count']
    
    if M == m:
        raise ZeroDivisionError('the series cannot be all the same')
        return
    
    if method == 'zscore':
        res = (dfcol - mu)/sig
            
    elif method == 'range_standard':
        res = (dfcol - mu) / (M - m)      
    
    elif method == 'range_regular':            
        res = (dfcol - m) / (M - m)
    
    elif method == 'div_max':
        res = dfcol / M
        
    elif method == 'min_div':
        res = m / dfcol
        
    elif method == 'weight':
        res = dfcol / (n * mu)
    
    elif method == 'sigmoid':
        res = 1 / (1 + np.exp(-dfcol))
    
    elif method == 'percentile':
        res = (dfcol.rank() - 1) / (max(dfcol.rank()) - 1)
        
    return res


# %%
# 补充缺失值
def fillna(dfcol,method ='mean',fix_value = 0):

    eligible_method = ['mean','median','fix_value','max','min']
    if method not in eligible_method:
        raise AttributeError('method must be one of %s' % ','.join(eligible_method))
        return
           
    stats = dfcol.describe()
    mu,q2 = stats.loc['mean'], stats.loc['50%']
    M,m = stats.loc['max'], stats.loc['min']
    
    if M == m:
        raise ZeroDivisionError('the series cannot be all the same')
        return
    
    if method == 'mean':
        res = dfcol.fillna(mu)
            
    elif method == 'median':
        res = dfcol.fillna(q2)      
    
    elif method == 'fix_value':            
        res = dfcol.fillna(fix_value)
    
    elif method == 'max':
        res = dfcol.fillna(M)
    
    elif method == 'min':
        res = dfcol.fillna(m)
        
    return res


# %%
#中性化
def neutralize(df_X,df_Y):

    if df_X.isnull().any():
        raise ValueError('please fill nan values before neutralization in df_X matrix')
        return
    if df_Y.isnull().any():
        raise ValueError('please fill nan values before neutralization in df_Y matrix')
        return
    if set(df_X.index) != set(df_Y.index):
        raise AttributeError('df_X,df_Y must have the same samples(stocks)')
        return
    
    res = df_Y.copy()
    for i in df_Y.columns:
        tmp_Y = df_Y[i]
        linreg = LinearRegression().fit(df_X,tmp_Y)        
        beta,intercept = linreg.coef_,linreg.intercept_
        resid = np.array(tmp_Y - np.dot(df_X,beta.T) - intercept)
        res[i] = resid.reshape(len(resid),)
    
    return res


# %%

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datavita.dpms.dpms_data import DpmsData
from datavita.core.common.auth import Auth

def get_col(data_code):
    access_key_id = '5rtxxn9xl43xhkzo'
    access_secret = 'e2c420d928d4bf8ce0ff2ec19b371514'
    auth = Auth(access_key_id, access_secret)
    dp = DpmsData(auth=auth, timeout=5, max_retries=1)
    df = pd.DataFrame()
    df = dp.GetEdb(dataCode=data_code, startDay=19900101, endDay=20251130, windId=False, pivot=False)
    res = df['data_value'].astype(float)
    return res

# 标准化
col = get_col(["TD00004764"])
col_sta_Z = standardize(col,method='zscore')
col_sta_mean = standardize(col,method='range_standard')
col_sta_minmax = standardize(col,method='range_regular')
col_sta_max = standardize(col,method='div_max')
col_sta_min = standardize(col,method='min_div')
col_sta_wei = standardize(col,method='weight')

# 异常值处理
col_win_sigmu = winsorize(col,method='nsigma_mean',nsigma = 3, niqr=1.5)
col_win_iqr = winsorize(col,method='boxplot',nsigma = 3, niqr=1.5)

col_win_med = winsorize(col,method='medcouple',nsigma = 3)
col_win_sigme = winsorize(col,method='nsigma_median',nsigma = 3)
col_win_qtl = winsorize(col,method='quantile',nsigma = 3)
col_win_median = winsorize(col,method='median',nsigma = 3)

