from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np
import os

date = ["2010-06-28", "2010-07-05"]
#list_stock = ["^DJI", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN"]
list_stock = ["TSLA", "MSFT"]

price = np.array([[2.0, 3.0],
                  [4.0, 5.0],
                  [6.0, 7.0]])

def process_data(prices, time_from_beg_vol = None):
    #calculate return of stock

    returns = (prices[1:] / prices[:-1]) - 1
    returns = np.vstack((np.nan * np.ones(returns.shape[1]), returns))
    print(returns)

    #calcul de vol
    n=1
    average_returns = np.nanmean(returns[-n:], axis=0)
    print(average_returns)






process_data(price)