#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 16:32:47 2022

@author: erinnsun
"""

import numpy as np
import pandas as pd

from datetime import timedelta
import pandas_market_calendars as mcal




def trade(start_date = '2021-09-03', 
          end_date = '2021-09-16', 
          cutoff = 0.25):
    
    path = '/Users/erinnsun/Desktop/alpaca/results/'
    
    symbols = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 
           'CSCO', 'CVX', 'GS', 'HD', 'HON', 
           'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 
           'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 
           'PG', 'TRV','UNH', 'CRM', 'VZ', 
           'V', 'WBA','WMT', 'DIS', 'DOW']
    
    # Retrieve the predicted data and the actual data
    data = {}
    for symbol in symbols:
        df = pd.read_csv(path + symbol + '.csv', index_col = 0, parse_dates = True)
        data[symbol] = df
    
    
    
    # Simulate trading
    nyse = mcal.get_calendar('NYSE') # get NYSE trading schedule
    trade_dates = list(nyse.schedule(start_date = start_date, end_date = end_date).index)
    trade_dates = [x.tz_localize('GMT') for x in trade_dates]
    
    for trade_date in trade_dates:
        
        df = pd.DataFrame()
        for symbol in data:
            df_tmp = data[symbol]
            df_tmp = df_tmp.loc[(df_tmp.index >= trade_date) & (df_tmp.index < trade_date + timedelta(days = 1))]
            df[symbol] = df_tmp['lr']
    
        for symbol in data:
            df[symbol + '-return'] = (df[symbol] - df[symbol].shift(1)) / df[symbol].shift(1)
        
        df.dropna(inplace = True)
        df.drop(symbols, axis = 1, inplace = True) # drop the non-return columns
        display(df)
        
        top_q = df.quantile(cutoff, axis = 1)
        
        display(top_q)
        
            
        