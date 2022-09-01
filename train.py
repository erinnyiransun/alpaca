#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 21:56:49 2022

@author: erinnsun
"""

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

from datetime import timedelta
import pandas_market_calendars as mcal

# from ml_models import train_lr # machine learning models


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor



'''
 Models
'''
def train_lr(X_train,y_train):
    
    
    
    lr_regressor = LinearRegression()
    model = lr_regressor.fit(X_train, y_train)
    
    return model


def train_lasso(X_train, y_train):
  
    lasso = Lasso()
    # scoring_method = 'r2'
    # scoring_method = 'explained_variance'
    # scoring_method = 'neg_mean_absolute_error'
    scoring_method = 'neg_mean_squared_error'
    # scoring_method = 'neg_mean_squared_log_error'
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20],
                  'tol': [1e-4, 1e-3, 1e-2, 1e-1]} # vary regularization 
    lasso_regressor = GridSearchCV(lasso, parameters, scoring=scoring_method, cv=3)
    lasso_regressor.fit(X_train, y_train)

    model = lasso_regressor.best_estimator_
    return model


def train_svm(X_train, y_train):

    svr = SVR()

    # param_grid_svm = {'C':[0.001, 0.01, 0.1, 1, 10],'gamma': [1e-7, 1e-4,0.001,0.1]}
    param_grid_svm = {'kernel': ('rbf','poly'), 'C':[0.1, 1, 10],'epsilon':[0.1,0.2]}

    # scoring_method = 'r2'
    # scoring_method = 'explained_variance'
    # scoring_method = 'neg_mean_absolute_error'
    scoring_method = 'neg_mean_squared_error'
    # scoring_method = 'neg_mean_squared_log_error'
    
    svm_regressor = GridSearchCV(estimator=svr, param_grid=param_grid_svm,
                                       cv=3, n_jobs=-1, scoring=scoring_method, verbose=0)

    svm_regressor.fit(X_train, y_train)
    model = svm_regressor.best_estimator_

    return model


def train_rf(X_train, y_train):
    
    randomforest_regressor = RandomForestRegressor(n_estimators = 50, min_samples_leaf = 100)
    model = randomforest_regressor.fit(X_train, y_train)

    
    return model








def prepare_train_data(df, 
                       start_date, end_date, 
                       lag,
                       feature_columns = ['open', 'high', 'low', 'close', 'volume'],
                       target_column = 'close'):
    '''

    Parameters
    ----------
    df : pandas.DataFrame
        raw alpaca stock data for an individual stock
    start_date : string (e.g. '2021-06-01T00:00:00Z')
        start date for the train window
    end_date : string
        end date for the train window
    lag : integer
        number of previous minutes to be used as features
    feature_columns : list of strings, optional
        The default is ['open', 'high', 'low', 'close', 'volume']
    target_column: string
        The column to be predicted

    Returns
    -------
    pandas.DataFrame
    processed stock data

    '''
    
   
    # Select the dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    data = df.loc[(df.index >= start_date) & (df.index < end_date)]
    
    # Select the features
    # data = data[feature_columns]
    
    # Add past data as features
    col_names = []
    for i in range(1, lag + 1):
        for col in feature_columns:
            col_name = col + '(t-' + str(i) + ')'
            data[col_name] = data[col].shift(i)
            col_names.append(col_name)
    
    # Drop records with incomplete features
    data.dropna(inplace = True)
    a = pd.to_datetime('2021-06-01T13:30:00Z')
    data = data.loc[data.index.time >= (a + timedelta(minutes = lag)).time()]
    
   
    # Prepare the dataset
    X = data[col_names].values
    y = data[target_column].values
    
    return (X, y, data.index)





def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import mean_squared_error
    #from sklearn.metrics import mean_squared_log_error

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import r2_score
    y_predict = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    explained_variance = explained_variance_score(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    return mse





def train(df,
          train_start_date, train_window_size,
          val_start_date, val_window_size,
          trade_start_date,
          n_windows,
          lag = 10,
          target_column = 'close'):
    '''
    

    Parameters
    ----------
    df : pandas.DataFrame
        stock price data for an individual stock
    train_start_date : string
        DESCRIPTION.
    train_window_size : integer
        number of days used for training
    trade_start_date : string
        DESCRIPTION.
    n_windows : integer
        number of trading days
    lag: integer, optional
        same as in 'prepare_train_data'

    Returns
    -------
    None.

    '''
    
    train_start_date = pd.to_datetime(train_start_date)
    val_start_date = pd.to_datetime(val_start_date)
    trade_start_date = pd.to_datetime(trade_start_date)
    
    train_start_dates = pd.date_range(train_start_date, train_start_date + timedelta(days = n_windows))
    val_start_dates = pd.date_range(val_start_date, val_start_date + timedelta(days = n_windows))
    trade_start_dates = pd.date_range(trade_start_date, trade_start_date + timedelta(days = n_windows))
    
    nyse = mcal.get_calendar('NYSE') # get NYSE trading schedule
    trade_dates = list(nyse.schedule(start_date = trade_start_date, end_date = trade_start_date + timedelta(days = n_windows - 1)).index)
    trade_dates = [x.tz_localize('GMT') for x in trade_dates]
    
    
    
    models = ['lr', 'rf']
    # models = ['lr', 'lasso', 'ridge', 'svm', 'gbm']
    df_predict = pd.DataFrame(columns = models, index = pd.date_range(start = trade_start_dates[0], 
                                                                      end = trade_start_dates[-1] + timedelta(days = 1), 
                                                                    freq = 'min'))
    # Daily retrain 
    for i in range(len(train_start_dates)):
        
        train_start_date = train_start_dates[i]
        train_end_date = train_start_date + timedelta(days = train_window_size)
        val_start_date = val_start_dates[i]
        val_end_date = val_start_date + timedelta(days = val_window_size)
        trade_start_date = trade_start_dates[i]
        
        if trade_start_date not in trade_dates: # not a trading day
            continue
        
        print('Trade date:', trade_start_date)
        
        # Prepare datasets
        X_train, y_train, _ = prepare_train_data(df, train_start_date, train_end_date, lag = lag, target_column = target_column)
        X_val, y_val, _ = prepare_train_data(df, val_start_date, val_end_date, lag = lag, target_column = target_column)
        X_trade, y_trade, times = prepare_train_data(df, trade_start_date, trade_start_date + timedelta(days = 1), lag = lag, target_column = target_column)
        
        
        # Train
        trained_models = {}
        for model in models:
            print(model)
            exec('trained_models[model] = train_' + model + '(X_train, y_train)')
        #lr_model = train_lr(X_train, y_train)
        
        
        
        
        # Validate
        scores = []
        for model in models:
            exec('scores.append(evaluate_model(trained_models[model], X_val, y_val))')
        
        current_min = scores[0]
        min_idx = 0
        for i in range(len(scores)):
            if scores[i] < current_min:
                min_idx = i
                current_min = scores[i]
        
        best_model = models[min_idx]
        
        
        
        # Trade
        for model in models:
            exec('df_predict.at[times, model] = ' + 'trained_models[model].predict(X_trade).flatten()')
        #y_trade_lr = lr_model.predict(X_trade).flatten()
        #df_predict.at[times, 'lr'] = y_trade_lr
        df_predict.at[times, 'actual'] = df[target_column][times]
        df_predict.at[times, 'best'] = trained_models[best_model].predict(X_trade).flatten()
    
    df_predict.dropna(inplace = True)
    return df_predict
        
        
        
      
        
      
        
      
   

if __name__ == '__main__':
    
    
    
    symbols = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 
           'CSCO', 'CVX', 'GS', 'HD', 'HON', 
           'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 
           'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 
           'PG', 'TRV','UNH', 'CRM', 'VZ', 
           'V', 'WBA','WMT', 'DIS', 'DOW']
    
    read_path = '/Users/erinnsun/Desktop/alpaca/data/'
    save_path = '/Users/erinnsun/Desktop/alpaca/results/'
    
    for symbol in symbols:
        path = read_path + symbol + '.csv'
        df = pd.read_csv(path, index_col = 0, parse_dates = True)
        df['return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        print(symbol)
        
        train_start_date = '2021-06-01T00:00:00Z'
        train_window_size = 30 + 31 + 14
        val_start_date = '2021-08-15T00:00:00Z'
        val_window_size = 17
        trade_start_date = '2021-09-03T00:00:00Z'
        n_windows = 14
        lag = 2
        
        
        df_predict = train(df, train_start_date, train_window_size,
                  val_start_date, val_window_size,
                  trade_start_date,
                  n_windows,
                  lag,
                  'return')
        
        df_predict.to_csv(save_path + symbol + '.csv')
    
    
    
    



        