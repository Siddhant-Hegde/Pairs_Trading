# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:48:07 2020

@author: ss466
"""

import yfinance as yf
import pandas as pd
import numpy as np 


time_period = '240mo' ###20 years of tick data
yahoo_option = 1
excel_option = 1 - yahoo_option

######splitting by sector
GICS_df = pd.read_csv('C:/Users/ss466/Documents/Trading Analysis/GICS split.csv', encoding= 'unicode_escape')

#GICS_df.columns[3] = 'GICS Sector'. For whatever reason hardcoding it results in an error
##dataframe of sector: list of tickers
ticks_by_GICS = {sector: list(GICS_df['Symbol'][GICS_df[GICS_df.columns[3]]==sector]) for sector in GICS_df[GICS_df.columns[3]].unique()}
print(ticks_by_GICS)

d_with_ticker_dfs = {}

def read_from_yahoo(ticks_by_GICS, yahoo_option):
    if yahoo_option:
        for sector in ticks_by_GICS:
            num = int(''.join(filter(str.isdigit, time_period))) ###digits in time_period
            d_with_ticker_dfs[sector] = pd.DataFrame(columns = ticks_by_GICS[sector], data = np.zeros((num,len(ticks_by_GICS[sector]))))
            ###Using closing prices, as is custom
            for tick in ticks_by_GICS[sector]:
                d_with_ticker_dfs[sector][tick] = yf.Ticker(ticks_by_GICS[sector][tick]).history(period = time_period)['Close']
        
        ### at least 80% non NaN values
        threshold = int(round(d_with_ticker_dfs[sector].shape[0] * 0.8))
        d_with_ticker_dfs[sector].dropna(axis = 1, thresh = threshold)
        d_with_ticker_dfs[sector].to_csv('C:/Users/ss466/Documents/Trading Analysis/DatabySector_{}'.format(sector))
        return d_with_ticker_dfs
    else:
        return None

def read_from_excel(ticks_by_GICS, excel_option):
    if excel_option:
        for sector in ticks_by_GICS:
            d_with_ticker_dfs[sector].read_csv('C:/Users/ss466/Documents/Trading Analysis/DatabySector_{}'.format(sector))
        return d_with_ticker_dfs
    else:
        return None
    
def which_function_to_run(*args, **kwargs):
    l = []
    for func in args:
        l.append(func(*kwargs[func.__name__]))
    return l
 
d_with_ticker_dfs = [func for func in which_function_to_run(read_from_yahoo, read_from_excel, read_from_yahoo = (ticks_by_GICS, yahoo_option), read_from_excel = (ticks_by_GICS, excel_option)) if func is not None]

print(d_with_ticker_dfs)