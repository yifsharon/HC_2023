# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:26:37 2020

@author: Wu Yiyang
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#去极值
def winsorize(dfcol,method='boxplot',nsigma = 3):
    '''

    Parameters
    ----------
    dfcol : a column in DataFrame
        The column that needs winsorization.
    method : STRING, optional
        Indicates the method of winsorization. The default is 'boxplot'.
    nsigma : NUMERIC, optional
        The threshold is nsigma times standard deviation. The default is 3.

    Raises
    ------
    AttributeError
        Methods should be one of 'boxplot','medcouple','nsigma_mean','nsigma_median','quantile','median'.

    Returns
    -------
    res : a column in DataFrame
        the factor after winsorization.
    upp : TYPE
        Winsorization upper limit.
    low : TYPE
        Winsorization lower limit.

    '''
    eligible_method = ['boxplot','medcouple','nsigma_mean','nsigma_median','quantile','median']
    if method not in eligible_method:
        raise AttributeError('method must be one of %s' % ','.join(eligible_method))
        return
           
    stats = dfcol.describe()
    mu,sig = stats.loc['mean'], stats.loc['std']
    q3,q2,q1 = stats.loc['75%'],stats.loc['50%'],stats.loc['25%']
    iqr = q3 - q1
    
    if method == 'boxplot':
        upp = q3 + 3 * iqr
        low = q1 - 3 * iqr
            
    elif method == 'medcouple':
        high_list = list(filter(lambda x:x>q2, list(dfcol)))
        low_list = list(filter(lambda x:x<q2, list(dfcol)))
        mc_list = [(x + y - 2*q2)/(x - y) for (x,y) in zip(high_list,low_list)]
        mc = np.median(mc_list)
        low = q1 - 1.5 * np.exp((-3.5 - 0.5 *(mc<0)) *mc) * iqr
        upp = q3 + 1.5 * np.exp((3.5 + 0.5 *(mc>=0)) *mc) * iqr            
    
    elif method == 'nsigma_mean':            
        upp = mu + nsigma * sig
        low = mu - nsigma * sig
    
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
    return res,upp,low   

#标准化
def standardize(dfcol,method='zscore'):
    '''   

    Parameters
    ----------
    dfcol : a column in DataFrame
        The column that needs standardization.
    method : STRING, optional
        The standardize method. The default is 'zscore'.

    Raises
    ------
    AttributeError
        Methods must be one of 'zscore','range_standard','range_regular','div_max','sigmoid','percentile'.
    ZeroDivisionError
        the factors cannot be identical.

    Returns
    -------
    res : a column in DataFrame
        The factor after standardization.

    '''
    eligible_method = ['zscore','range_standard','range_regular','div_max','sigmoid','percentile']
    if method not in eligible_method:
        raise AttributeError('method must be one of %s' % ','.join(eligible_method))
        return
           
    stats = dfcol.describe()
    mu,sig = stats.loc['mean'], stats.loc['std']
    M,m = stats.loc['max'], stats.loc['min']
    
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
    
    elif method == 'sigmoid':
        res = 1 / (1 + np.exp(-dfcol))
    
    elif method == 'percentile':
        res = (dfcol.rank() - 1) / (max(dfcol.rank()) - 1)
        
    return res


# 补充缺失值
def fillna(dfcol,method ='mean',fix_value = 0):
    '''

    Parameters
    ----------
    dfcol : a column in DataFrame
        The column that needs fillna.
    method : STRING, optional
        fiilna methods. The default is 'mean'.
    fix_value : NUMERIC, optional
        if 'fix_value' method is selected, set the fix_value to fillna. The default is 0.

    Raises
    ------
    AttributeError
        Methods must be one of 'mean','median','fix_value','max','min'.
    ZeroDivisionError
        the factors cannot be identical.

    Returns
    -------
    res : a column in DataFrame
        The factor after nan filling.

    '''
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

#中性化
def neutralize(df_X,df_Y):
    '''

    Parameters
    ----------
    df_X : pandas DataFrame
        all factors to neutralize on organized by each factor in columns and each sample in rows.
    df_Y : pandas DataFrame
        all factors to neutralize organized by each factor in columns and each sample in rows.

    Raises
    ------
    ValueError
        df_X,df_Y cannot contain nan.
    AttributeError
        df_X,df_Y must have the same samples.

    Returns
    -------
    res : pandas DataFrame
        Factors after neutralization organized by each factor in columns and each sample in rows.

    '''
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