#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 16:32:47 2022

@author: erinnsun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from datetime import timedelta
import pandas_market_calendars as mcal




def get_return_dic(predict_return = True):
    
    path = '/Users/erinnsun/Desktop/alpaca/results/'
    
    symbols = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 
           'CSCO', 'CVX', 'GS', 'HD', 'HON', 
           'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 
           'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 
           'PG', 'TRV','UNH', 'CRM', 'VZ', 
           'V', 'WBA','WMT', 'DIS', 'DOW']
    
    # Retrieve the predicted data and the real data
    data = {}
    for symbol in symbols:
        df = pd.read_csv(path + symbol + '.csv', index_col = 0, parse_dates = True)
        
        if not predict_return: # if the target is the stock price itself
            for col in df.columns:
                df[col] = (df[col] - df[col].shift(1)) / df[col]
            start_time = df.index[1].time()
            df = df.loc[df.index.time == start_time] # drop those lines without return values
            
        data[symbol] = df
    
    return data
         

    
    
    


def trade(start_date = '2021-09-03', 
          end_date = '2021-09-16', 
          cutoff = 0.75,
          predict_return = True):
    
    
    # Get prediction results
    data = get_return_dic(predict_return)
    
    
    # Get trading days
    nyse = mcal.get_calendar('NYSE') # get NYSE trading schedule
    trade_dates = list(nyse.schedule(start_date = start_date, end_date = end_date).index)
    trade_dates = [x.tz_localize('GMT') for x in trade_dates]
    
    
    # Prepare the portfolio return table
    portfolio_return_dic = {}
    equal_portfolio_return_dic = {} # baseline
    
    
    # Simulated Trading
    for trade_date in trade_dates:
        
        # Combine all stocks' data
        df = pd.DataFrame()
        df_actual = pd.DataFrame()
        for symbol in data:
            df_tmp = data[symbol]
            df_tmp = df_tmp.loc[(df_tmp.index >= trade_date) & (df_tmp.index < trade_date + timedelta(days = 1))]
            df[symbol + '-return'] = df_tmp['rf'] # to be changed to ['best']
            df_actual[symbol + '-return'] = df_tmp['actual']
        
        # Calculate the cutoff
        top_q = df.quantile(cutoff, axis = 1)
        # top_q.clip(lower = 0, inplace = True)
        
        long_dict = {}
        for i in range(df.shape[0]):
            long_dict[df.index[i]] = df.iloc[i][df.iloc[i] > top_q[i]]
            # print(long_dict[df.index[i]], top_q[i])
            # print(df_actual.iloc[i][df.iloc[i] > top_q[i]])
        
        for i in range(df.shape[0]):
            
            # Baseline strategy
            returns = df_actual.loc[df_actual.index == df.index[i]]
            equal_portfolio_return_dic[df.index[i]] = (returns.values * 1/30).sum()
            
            if long_dict[df.index[i]].shape[0] == 0: 
                portfolio_return_dic[df.index[i]] = 0
                continue
            
            # Long-only strategy
            long_normalize_weight = long_dict[df.index[i]] / sum(long_dict[df.index[i]].values)
            a = long_dict[df.index[i]].index # list of companie to be traded
            long_tic_return = df_actual[a].loc[df_actual.index == df.index[i]]
            long_return_table = long_tic_return * long_normalize_weight 
            portfolio_return_dic[df.index[i]] = long_return_table.values.sum()
            
    
    
    df_portfolio_return = pd.DataFrame.from_dict(portfolio_return_dic, orient='index')
    df_portfolio_return.columns = ['return']
    df_portfolio_return.index.names = ['time']
    
    equal_df = pd.DataFrame.from_dict(equal_portfolio_return_dic, orient='index')
    equal_df.columns = ['return']
    equal_df.index.names = ['time']
    
    return (df_portfolio_return['return'], equal_df['return'])





def plot_return(df_portfolio_return, equal_return):
    
    
    # convert indices to strings
    times = []
    xticks = []
    current = pd.to_datetime('2021-09-02')
    
    for time in list(df_portfolio_return.index):
        times.append(time.isoformat())
        if time.date != current.date:
            xticks.append(time.isoformat())
    df_portfolio_return.index = times
    equal_return.index = times
    
    
    
    fig, ax = plt.subplots(figsize=(15,10))
    #majors = xticks
    #ax.xaxis.set_major_locator(ticker.FixedLocator(majors))
    
    lr = ((df_portfolio_return + 1).cumprod() - 1).plot(ax = ax, c = 'b',label = 'random forest')
    equal = ((equal_return + 1).cumprod() - 1).plot(ax = ax, c = 'r',label = 'equal')
    
    #ax.set_xticks(xticks)
    plt.xticks(rotation=45)
    plt.legend()
    plt.title('Cumulative Return', {'size':16})
    plt.show()
    
            



def plot_stock_price(symbol,
                     start_date = '2021-09-03',
                    end_date = '2021-09-16',
                    ):
    
    path = '/Users/erinnsun/Desktop/alpaca/results/lr_results/'
    
    symbols = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 
           'CSCO', 'CVX', 'GS', 'HD', 'HON', 
           'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 
           'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 
           'PG', 'TRV','UNH', 'CRM', 'VZ', 
           'V', 'WBA','WMT', 'DIS', 'DOW']
    
    
    
    df = pd.read_csv(path + symbol + '.csv', index_col = 0)
    a = df['actual']
    df.plot()
        