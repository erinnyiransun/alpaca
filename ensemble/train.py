import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append('../')
from models import *
from build_dataset import build_dataset






'''
1. Utility Functions
'''

def prepare_train_data(df, start_time, end_time,
                       feature_columns,
                       target_column
                       ):
  
  start_time = pd.to_datetime(start_time)
  end_time = pd.to_datetime(end_time)
  df['time'] = pd.to_datetime(df['time'])

  # select the range of data
  df = df.loc[(df['time'] >= start_time) & (df['time'] <= time)]
  
  # prepare return values
  X_train = pd.DataFrame(columns = feature_columns)
  y_train = pd.DataFrame(columns = [target_column])
  df_scaler = pd.DataFrame(columns = ['scaler', 'ticker'])

  companies = df['symbol'].unique()
  for cpn in companies:
    df_temp = df[df['symbol'] == cpn]
    y_temp = df_temp[[target_column]]
    X_temp = df_temp[feature_columns]

    # normalize
    scaler = MinMaxScaler()
    X_temp = pd.DataFrame(data = scaler.fit_transform(X_temp), columns = feature_columns)
    df_scaler.append({"scaler": scaler, "ticker": cpn}, ignore_index = True)

    # concatenate to training data
    X_train = pd.concat([X_train, X_temp], axis = 0)
    y_train = pd.concat([y_train, y_temp], axis = 0)
  
  df_scaler.index = df_scaler.githubUser
  return (X_train.values, y_train.values.ravel(), df_scaler)



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


def append_return_table(df_predict, unique_datetime, y_trade_return, trade_tic, current_index):
    tmp_table = pd.DataFrame(columns=trade_tic)
    tmp_table = tmp_table.append(pd.Series(y_trade_return, index=trade_tic), ignore_index=True)
    df_predict.loc[unique_datetime[current_index]][tmp_table.columns] = tmp_table.loc[0]





'''
2. Training

'''
def train(df, # output of build_dataset()
            train_start_time,
            n_windows, # number of windows in the backtest
            train_window, # training window size (minutes)
            val_window, # validation window size (minutes)
            n_steps,
            result_path = '../../results/'): # number of timesteps (
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  s) in each sample 
    
    train_start_time = pd.to_datetime(train_start_time)
    train_start_times = [train_start_time + relativedelta(minutes = x) for x in range(n_windows)]
    val_start_times = [x + relativedelta(minutes = train_window) for x in train_start_times]
    trade_times = [x + relativedelta(minutes = val_window) for x in val_start_times]
    print("Last trade time:", trade_times[-1])
    unique_ticker = set(df['ticker'])
    unique_datetime = trade_times
    
    df_predict_lr = pd.DataFrame(columns=unique_ticker, index=unique_datetime)
    df_predict_lasso = pd.DataFrame(columns=unique_ticker, index=unique_datetime)
    df_predict_ridge = pd.DataFrame(columns=unique_ticker, index=unique_datetime)
    df_predict_rf = pd.DataFrame(columns=unique_ticker, index=unique_datetime)
    df_predict_svm= pd.DataFrame(columns=unique_ticker, index=unique_datetime)
    df_predict_gbm = pd.DataFrame(columns=unique_ticker, index=unique_datetime)
    df_predict_ada = pd.DataFrame(columns=unique_ticker, index=unique_datetime)
    df_predict_lstm = pd.DataFrame(columns=unique_ticker, index=unique_datetime)
    
    df_predict_best = pd.DataFrame(columns=unique_ticker, index=unique_datetime)
    df_best_model_name = pd.DataFrame(columns=['model_name'], index=unique_datetime)
    evaluation_record = []
    
    
    for i in range(len(unique_datetime)):
      train_start_time = train_start_times[i]
      val_start_time = val_start_times[i]
      trade_time = unique_datetime[i]
    
      X_train, y_train, df_scaler = prepare_train_data(df, train_start_time, train_start_time + relativedelta(minutes=train_window), 
                                                       n_steps, feature_columns)
      X_train_lstm = np.reshape(X_train, (X_train.shape[0], -1, len(feature_columns))) # (samples, timesteps, features)
      print(X_train_lstm.shape, y_train.shape)
    
      X_val, y_val, _ = prepare_val_data(df, val_start_time, val_start_time + relativedelta(minutes=val_window), df_scaler, n_steps, feature_columns)
      X_val_lstm = np.reshape(X_val, (X_val.shape[0], -1, len(feature_columns),))
    
      X_trade, y_trade, trade_tic = prepare_val_data(df, trade_time, trade_time, df_scaler, n_steps, feature_columns)
      X_trade_lstm = np.reshape(X_trade, (X_trade.shape[0], -1, len(feature_columns)))
    
       # train the models
      lr_model = train_linear_regression(X_train, y_train)
      lasso_model = train_lasso(X_train, y_train)
      ridge_model = train_ridge(X_train, y_train)
      rf_model = train_random_forest(X_train, y_train)
      svm_model = train_svm(X_train,y_train)
      gbm_model = train_gbm(X_train, y_train)
      ada_model = train_ada(X_train, y_train)
      lstm_model = train_lstm(X_train_lstm, y_train)
    
      # validation 
      lr_eval = evaluate_model(lr_model, X_val, y_val)
      lasso_eval = evaluate_model(lasso_model, X_val, y_val)
      ridge_eval = evaluate_model(ridge_model, X_val, y_val)
      rf_eval = evaluate_model(rf_model, X_val, y_val) 
      svm_eval = evaluate_model(svm_model, X_val, y_val)
      gbm_eval = evaluate_model(gbm_model, X_val, y_val)
      ada_eval = evaluate_model(ada_model, X_val, y_val)
      lstm_eval = evaluate_model(lstm_model, X_val_lstm, y_val)
    
            
      # trade
      y_trade_lr = lr_model.predict(X_trade).flatten()
      y_trade_lasso = lasso_model.predict(X_trade).flatten()
      y_trade_ridge = ridge_model.predict(X_trade).flatten()
      y_trade_rf = rf_model.predict(X_trade).flatten()
      y_trade_svm = svm_model.predict(X_trade).flatten()
      y_trade_gbm = gbm_model.predict(X_trade).flatten()
      y_trade_ada = ada_model.predict(X_trade).flatten()
      y_trade_lstm = lstm_model.predict(X_trade_lstm).flatten()
    
      # prepare evaluation data and predicted return
      eval_data = [[lr_eval, y_trade_lr], 
                        [lasso_eval, y_trade_lasso],
                         [ridge_eval, y_trade_ridge],
                         [rf_eval, y_trade_rf], 
                         [svm_eval,y_trade_svm],
                         [gbm_eval,y_trade_gbm],                     
                         [ada_eval,y_trade_ada],
                        [lstm_eval,y_trade_lstm]
                        ]
    
      eval_table = pd.DataFrame(eval_data, columns=['model_eval', 'model_predict_return'],
                                      index=['lr', 
                                             'lasso',
                                             'ridge',
                                             'rf', 
                                             'svm',
                                             'gbm',
                                             'ada', 
                                             'lstm'])  
      evaluation_record.append((trade_time, eval_table))
    
      # lowest error score model
      y_trade_best = eval_table.model_predict_return.values[eval_table.model_eval == eval_table.model_eval.min()][0]
      best_model_name = eval_table.index.values[eval_table.model_eval == eval_table.model_eval.min()][0]
    
      df_best_model_name.loc[trade_time] = best_model_name
    
      # Prepare Predicted Return table
      append_return_table(df_predict_lr, unique_datetime, y_trade_lr, trade_tic, current_index=i)
      append_return_table(df_predict_lasso, unique_datetime, y_trade_lasso, trade_tic, current_index=i)
      append_return_table(df_predict_ridge, unique_datetime, y_trade_ridge, trade_tic, current_index=i)
      append_return_table(df_predict_rf, unique_datetime, y_trade_rf, trade_tic, current_index=i)
      append_return_table(df_predict_svm, unique_datetime, y_trade_svm, trade_tic, current_index=i)
      append_return_table(df_predict_gbm, unique_datetime, y_trade_gbm, trade_tic, current_index=i)
      append_return_table(df_predict_ada, unique_datetime, y_trade_ada, trade_tic, current_index=i)
      append_return_table(df_predict_lstm, unique_datetime, y_trade_lstm, trade_tic, current_index=i)
      append_return_table(df_predict_best, unique_datetime, y_trade_best, trade_tic, current_index=i)
    
      print("Trade time:", trade_time)
    
    
    df_predict_lr.to_csv(result_path + "lr.csv")
    df_predict_lasso.to_csv(result_path + "lasso.csv")
    df_predict_ridge.to_csv(result_path + "ridge.csv")
    df_predict_rf.to_csv(result_path + "rf.csv")
    df_predict_svm.to_csv(result_path + "svm.csv")
    df_predict_gbm.to_csv(result_path + "gbm.csv")
    df_predict_ada.to_csv(result_path + "ada.csv")
    df_predict_lstm.to_csv(result_path + "lstm.csv")
    df_predict_best.to_csv(result_path + "best.csv")
    
    return (trade_times, unique_ticker) # for visualization purposes




    
    
