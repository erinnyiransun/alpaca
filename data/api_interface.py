import alpaca_trade_api as api
from alpaca_trade_api.rest import TimeFrame
import datetime as dt
from dateutil import parser


'''
positions = alpaca.list_positions()
for position in positions:
    print(position)
'''


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



def get_minute_data(alpaca, symbol, time_start = '2021-06-01T00:00:00Z', time_end = '2021-09-17T00:00:00Z'):
    

    bars = alpaca.get_bars(symbol, TimeFrame.Minute, start = time_start, end = time_end)
    return bars.df
    


if __name__ == '__main__':
  
  
    API_KEY = 'PKOTJX0NGI6214R71VL1'
    API_SECRET = 'jS23GtIYLipaX4YJr7IhBOCWm9apvV7gqEtKhFq4'
    BASE_URL = 'https://paper-api.alpaca.markets'

    alpaca = api.REST(API_KEY, API_SECRET, BASE_URL)
    account = alpaca.get_account()

    
    # Dow Jones 30 
    symbols = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 
           'CSCO', 'CVX', 'GS', 'HD', 'HON', 
           'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 
           'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 
           'PG', 'TRV','UNH', 'CRM', 'VZ', 
           'V', 'WBA','WMT', 'DIS', 'DOW']
    
    
    data = []
    for symbol in symbols:
      data.append(get_minute_data(alpaca, symbol))
      
      
    
