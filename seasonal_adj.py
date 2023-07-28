# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 00:40:52 2023

@author: YIFEI YAO
"""

from statsmodels.tsa.seasonal import seasonal_decompose
from datavita.core.common.auth import Auth
from datavita.dpms.dpms_data import DpmsData
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


def get_tsInfo(data_code):
    """
    获取数据值
    @param data_code: 数据编码
    @return: 时间序列
    """
    access_key_id = 'j49k37x4l43xgexj'
    access_secret = 'd3d9446802a44259755d38e6d163e820'
    auth = Auth(access_key_id, access_secret)
    dp = DpmsData(auth=auth, timeout=5, max_retries=1)
    df = dp.GetEdb(dataCode=data_code, startDay=19900101,
                   endDay=20251130, windId=False, pivot=False)
    df.set_index(pd.to_datetime(df['data_day'],
                 format='%Y%m%d'), drop=True, inplace=True)
    df = df.sort_index()
    ts = df['data_value'].astype(float)
    return ts


def seas_expanding_decompose(ts, datacode, window_size):
    """
    季节性调整1：拆分
    @param ts: 时间序列
    @param datacode: 数据编码
    @param window_size: 初次窗口大小
    @return: 模型结果
    """
    trend_add, trend_mul = pd.Series(), pd.Series()
    seasonal_add, seasonal_mul = pd.Series(), pd.Series()
    resid_add, resid_mul = pd.Series(), pd.Series()
    seasadj_add, seasadj_mul = pd.Series(), pd.Series()
    for i in range(window_size, len(ts)+1):
        window = ts.iloc[:i]
        dcp_add = seasonal_decompose(
            window, model='additive', period=12, two_sided=False)
        dcp_add_seasadj = dcp_add.observed - dcp_add.seasonal
        trend_add = trend_add.combine_first(dcp_add.trend)
        resid_add = resid_add.combine_first(dcp_add.resid)
        seasonal_add = seasonal_add.combine_first(dcp_add.seasonal)
        seasadj_add = seasadj_add.combine_first(dcp_add_seasadj)
        if (ts > 0).all():
            dcp_mul = seasonal_decompose(
                window, model='multiplicative', period=12, two_sided=False)
            dcp_mul_seasadj = dcp_mul.observed - dcp_mul.seasonal
            trend_mul = trend_mul.combine_first(dcp_mul.trend)
            resid_mul = resid_mul.combine_first(dcp_mul.resid)
            seasonal_mul = seasonal_mul.combine_first(dcp_mul.seasonal)
            seasadj_mul = seasadj_mul.combine_first(dcp_mul_seasadj)
    df = pd.DataFrame({'data_code': datacode, 'observed': ts}, index=ts.index)
    df['trend'], df['seasonal'] = trend_add, seasonal_add
    df['resid'], df['seasadj'] = resid_add, seasadj_add
    df['type'] = 'add'
    if (ts > 0).all():
        df1 = pd.DataFrame(
            {'data_code': datacode, 'observed': ts}, index=ts.index)
        df1['trend'], df1['seasonal'] = trend_mul, seasonal_mul
        df1['resid'], df1['seasadj'] = resid_mul, seasadj_mul
        df1['type'] = 'mul'
        df = pd.concat([df, df1])
    return df




def seas_plot_decompose(df):
    """
    图像1：拆分
    @param df: 模型结果
    @output: 图像
    """
    df1 = df.loc[df['type'] == 'add']
    fig1, axs1 = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    axs1[0, 0].plot(df1.index, df1['observed'], label='observed')
    axs1[0, 0].plot(df1.index, df1['seasadj'], label='seasadj')
    axs1[0, 0].legend(loc='upper left')
    axs1[0, 1].plot(df1.index, df1['seasonal'], label='seasonal')
    axs1[0, 1].legend(loc='upper left')
    axs1[1, 0].plot(df1.index, df1['trend'], label='trend')
    axs1[1, 0].legend(loc='upper left')
    axs1[1, 1].plot(df1.index, df1['resid'], label='resid')
    axs1[1, 1].legend(loc='upper left')
    fig1.suptitle('decompose_add  %s' % df1['data_code'][0])
    check_exist_and_makedir('plots_dcp_add', path_output)
    #plt.savefig(r'./output/plots_dcp_add/%s.png' % df1['data_code'][0])
    plt.show()
    if 'mul' in df['type'].values:
        df2 = df.loc[df['type'] == 'mul']
        fig2, axs2 = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
        axs2[0, 0].plot(df2.index, df2['observed'], label='observed')
        axs2[0, 0].plot(df2.index, df2['seasadj'], label='seasadj')
        axs2[0, 0].legend(loc='upper left')
        axs2[0, 1].plot(df2.index, df2['seasonal'], label='seasonal')
        axs2[0, 1].legend(loc='upper left')
        axs2[1, 0].plot(df2.index, df2['trend'], label='trend')
        axs2[1, 0].legend(loc='upper left')
        axs2[1, 1].plot(df2.index, df2['resid'], label='resid')
        axs2[1, 1].legend(loc='upper left')
        fig2.suptitle('decompose_mul  %s' % df2['data_code'][0])
        check_exist_and_makedir('plots_dcp_mul', path_output)
        #plt.savefig(r'./output/plots_dcp_mul/%s.png' % df2['data_code'][0])
        plt.show()


def seas_expanding_x13arima(ts, datacode, window_size):
    """
    季节性调整2：x13arima
    @param ts: 时间序列
    @param datacode: 数据编码
    @param window_size: 初次窗口大小
    @return: 模型结果
    """
    trend, seasonal = pd.Series(), pd.Series()
    resid, seasadj = pd.Series(), pd.Series()
    for i in range(window_size, len(ts)+1):
        window = ts.iloc[:i]
        try:
            x13 = sm.tsa.x13_arima_analysis(window)
        except:
            continue
        if (x13.irregular > 0).all():
            x13_resid = x13.observed - (x13.observed / x13.irregular)
        else:
            x13_resid = x13.irregular
        x13_seasonal = x13.observed - x13.seasadj
        trend = pd.concat([trend, x13.trend[[-1]]])
        resid = pd.concat([resid, x13_resid[[-1]]])
        seasadj = pd.concat([seasadj, x13.seasadj[[-1]]])
        seasonal = pd.concat([seasonal, x13_seasonal[[-1]]])
    df = pd.DataFrame({'data_code': datacode, 'observed': ts}, index=ts.index)
    df['trend'], df['seasonal'] = trend, seasonal
    df['resid'], df['seasadj'] = resid, seasadj
    return df


def seas_plot_x13arima(df):
    """
    图像2：x13arima
    @param df: 模型结果
    @output: 图像
    """
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    axs[0, 0].plot(df.index, df['observed'], label='observed')
    axs[0, 0].plot(df.index, df['seasadj'], label='seasadj')
    axs[0, 0].legend(loc='upper left')
    axs[0, 1].plot(df.index, df['seasonal'], label='seasonal')
    axs[0, 1].legend(loc='upper left')
    axs[1, 0].plot(df.index, df['trend'], label='trend')
    axs[1, 0].legend(loc='upper left')
    axs[1, 1].plot(df.index, df['resid'], label='resid')
    axs[1, 1].legend(loc='upper left')
    fig.suptitle('X13ARIMA_method  %s' % df['data_code'][0])
    check_exist_and_makedir('output_plots_x13arima',  path_output)
    #plt.savefig(r'./output/output_plots_x13arima/%s.png' % df['data_code'][0])
    plt.show()


def seas_rolling_SMA(ts, datacode, window_size):
    """
    季节性调整3：SMA
    @param ts: 时间序列
    @param datacode: 数据编码
    @param window_size: 滚动窗口大小
    @return: 模型结果
    """
    seasadj = ts.rolling(window=window_size).mean()
    seasonal = ts - seasadj
    df = pd.DataFrame({'data_code': datacode, 'observed': ts}, index=ts.index)
    df['seasonal'], df['seasadj'] = seasonal, seasadj
    return df


def seas_plot_SMA(df):
    """
    图像3：SMA
    @param df: 模型结果
    @output: 图像
    """
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    axs[0].plot(df.index, df['observed'], label='observed')
    axs[0].plot(df.index, df['seasadj'], label='seasadj')
    axs[0].legend(loc='upper left')
    axs[1].plot(df.index, df['seasonal'], label='seasonal')
    axs[1].legend(loc='upper left')
    fig.suptitle('SMA_method  %s' % df['data_code'][0])
    check_exist_and_makedir('output_plots_SMA',  path_output)
    #plt.savefig(r'./output/output_plots_SMA/%s.png' % df['data_code'][0])
    plt.show()


def seas_expanding_HPfilter(ts, datacode, window_size):
    """
    季节性调整4：HP滤波
    @param ts: 时间序列
    @param datacode: 数据编码
    @param window_size: 初次窗口大小
    @return: 模型结果
    """
    seasonal, seasadj = pd.Series(), pd.Series()
    for i in range(window_size, len(ts)+1):
        window = ts.iloc[:i]
        cycle, trend = sm.tsa.filters.hpfilter(window, lamb=129600)
        seasonal = seasonal.combine_first(cycle)
        seasadj = seasadj.combine_first(trend)
    df = pd.DataFrame({'data_code': datacode, 'observed': ts}, index=ts.index)
    df['seasonal'], df['seasadj'] = seasonal, seasadj
    return df


def seas_plot_HPfilter(df):
    """
    图像4：HP滤波
    @param df: 模型结果
    @output: 图像
    """
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    axs[0].plot(df.index, df['observed'], label='observed')
    axs[0].plot(df.index, df['seasadj'], label='seasadj')
    axs[0].legend(loc='upper left')
    axs[1].plot(df.index, df['seasonal'], label='seasonal')
    axs[1].legend(loc='upper left')
    fig.suptitle('HPfilter_method  %s' % df['data_code'][0])
    check_exist_and_makedir('output_plots_HPfilter',  path_output)
    #plt.savefig(r'./output/output_plots_HPfilter/%s.png' % df['data_code'][0])
    plt.show()


path = os.path.dirname(__file__)
path_output = path + '/output/'


def check_exist_and_makedir(filename, path):
    """
    检查路径创建文件夹
    @param filename: 文件夹名称
    @param path: 路径
    @output: 创建文件夹
    """
    files = os.listdir(path)
    current_file = [x for x in files]
    if filename in current_file:
        pass
    else:
        folder_path = os.path.join(path, filename)
        os.mkdir(folder_path)


# %%
if __name__ == '__main__':
    # 获取季节性数据编码
    datalist = pd.read_excel(r'./input/data_seas.xlsx')
    datalist = datalist.drop(columns='Unnamed: 0').reset_index(drop=True)

    # 每种方法建立一个dataframe存储该模型汇总的结果
    df_dcp, df_x13 = pd.DataFrame(), pd.DataFrame()
    df_sma, df_hp = pd.DataFrame(), pd.DataFrame()

    # 对每一个datacode进行处理
    for i in range(0,11):
        datacode = datalist['data_code'][i]
        print(i)
        print(datacode)

        # 获取数据值并填充变频为月度
        ts = get_tsInfo(datacode)
        #ts = ts.resample('M').mean().ffill()
        ts = ts.resample('M').last().dropna(how='any')

        # 得到当前数据的decompose模型结果，存储图像，拼接结果到decompose汇总表中
        df_dcp_temp = seas_expanding_decompose(ts, datacode, 24)
        seas_plot_decompose(df_dcp_temp)
        df_dcp = pd.concat([df_dcp, df_dcp_temp])

        # 得到当前数据的X13ARIMA模型结果，存储图像，拼接结果到X13ARIMA汇总表中
        df_x13_temp = seas_expanding_x13arima(ts, datacode, 36)
        seas_plot_x13arima(df_x13_temp)
        df_x13 = pd.concat([df_x13, df_x13_temp])

        # 得到当前数据的SMA模型结果，存储图像，拼接结果到SMA汇总表中
        df_sma_temp = seas_rolling_SMA(ts, datacode, 12)
        seas_plot_SMA(df_sma_temp)
        df_sma = pd.concat([df_sma, df_sma_temp])

        # 得到当前数据的HPfilter模型结果，存储图像，拼接结果到HPfilter汇总表中
        df_hp_temp = seas_expanding_HPfilter(ts, datacode, 24)
        seas_plot_HPfilter(df_hp_temp)
        df_hp = pd.concat([df_hp, df_hp_temp])

    df_dcp.to_excel(r'./output/results_seas_adj_decompose.xlsx')
    df_x13.to_excel(r'./output/results_seas_adj_x13arima.xlsx')
    df_sma.to_excel(r'./output/results_seas_adj_SMA.xlsx')
    df_hp.to_excel(r'./output/results_seas_adj_HPfilter.xlsx')
