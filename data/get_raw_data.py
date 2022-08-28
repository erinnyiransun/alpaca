#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:27:50 2022

@author: erinnsun
"""

import alpaca_trade_api as api
from alpaca_trade_api.rest import TimeFrame
import datetime as dt
from dateutil import parser


import pandas as pd
import pandas_market_calendars as mcal


def buy(alpaca, symbol = 'BTC/USD', qty = 1):
    
    order = alpaca.submit_order(
        symbol, 
        qty = qty, 
        time_in_force = 'gtc')
    
    print('Buyed')
    return




def sell(alpaca, symbol = 'BTC/USD', qty = 1):
    
    order = alpaca.submit_order(
        symbol = symbol,
        qty = qty,
        side = "sell",
        time_in_force = 'gtc')
    
    print('Selled')
    return



def get_minute_data(alpaca, symbol, time_start = '2021-06-01T00:00:00Z', time_end = '2021-09-20T00:00:00Z'):
    

    bars = alpaca.get_bars(symbol, TimeFrame.Minute, start = time_start, end = time_end)
    df = bars.df
    
    trade_start_time = '13:30' # GMT +0 time; NYSE and Nasdaq hours
    trade_end_time = '20:00'
    df = df.between_time(start_time = trade_start_time, end_time = trade_end_time)
    
    # get trade days
    nyse = mcal.get_calendar('NYSE') # get NYSE trading schedule
    trade_dates = list(nyse.schedule(start_date = '2021-06-01', end_date = '2021-09-19').index)
    trade_dates = [x.date() for x in trade_dates]
    
    complete_df = pd.DataFrame(index = pd.date_range(time_start, time_end, freq = 'min'), columns = df.columns)
    complete_df['day'] = complete_df.index.date
    complete_df = complete_df.loc[complete_df.day.isin(trade_dates)]
    
    complete_df.drop('day', axis = 1, inplace = True)
    complete_df = complete_df.between_time(start_time = trade_start_time, end_time = trade_end_time)
    
    complete_df.at[df.index, df.columns] = df.values
    complete_df.fillna(method = 'ffill', inplace = True)
    
    return complete_df
    



if __name__ == '__main__':
    

    API_KEY = 'PKOTJX0NGI6214R71VL1'
    API_SECRET = 'jS23GtIYLipaX4YJr7IhBOCWm9apvV7gqEtKhFq4'
    BASE_URL = 'https://paper-api.alpaca.markets'

    alpaca = api.REST(API_KEY, API_SECRET, BASE_URL)

    
    symbols = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 
           'CSCO', 'CVX', 'GS', 'HD', 'HON', 
           'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 
           'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 
           'PG', 'TRV','UNH', 'CRM', 'VZ', 
           'V', 'WBA','WMT', 'DIS', 'DOW']
    
    
   
    
    
    
    save_path = '/Users/erinnsun/Desktop/alpaca/data/'
    for symbol in symbols:
        print(symbol)
        df = get_minute_data(alpaca, symbol)
        print(df.shape)
        df.to_csv(save_path + symbol + '.csv')

    
    
    
    
