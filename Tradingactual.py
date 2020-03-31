# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:48:07 2020

@author: ss466

This file is used to obtains pairs within the GICS Sectors of the S&P.
These pairs are chosen based on their degree of correlation and if they are
Cointegrated based on the Johansen test. 

Once selected their price ratios are calculated and histrocial std's determined
to see when would be an ideal time to enter into the pairs trade.  
"""

import yfinance as yf
import pandas as pd
import numpy as np 
from tests import ADF
from tests import get_johansen
import statistics

time_period = '240mo' ###20 years of tick data
no_pairs = 4 ###4 pairs for each sector

###need to optimize these parameters
observe_pair_period = 10 ###no of days pair is observed (i.e. std, mean to see if its worth getting into)
stay_in_pair_period = 30 ###no of days in the pair
total_period = observe_pair_period + stay_in_pair_period

####Do we want to scrape the data from yahoo or used excel files that contain data from when the yahoo data 
####was previously scraped
yahoo_option = 0 ##get yahoo data
excel_option = 1 - yahoo_option

###purely used to get dimensions of dataframes for each sector
sandp = yf.Ticker('^GSPC').history(time_period)

######splitting by sector
GICS_df = pd.read_csv('C:/Git stuff/Pairs_Trading/GICS split.csv', encoding= 'unicode_escape')

#GICS_df.columns[3] = 'GICS Sector'. For whatever reason hardcoding it results in an error
##dataframe of sector: list of tickers
ticks_by_GICS = {sector: list(GICS_df['Symbol'][GICS_df[GICS_df.columns[3]]==sector]) for sector in GICS_df[GICS_df.columns[3]].unique()}

d_with_ticker_dfs = {sector: pd.DataFrame(index = sandp.index, columns = ticks_by_GICS[sector], data = np.zeros((sandp.shape[0] ,len(ticks_by_GICS[sector])))) for sector in ticks_by_GICS}

def read_from_yahoo(ticks_by_GICS, yahoo_option):
    if yahoo_option:
        for sector in ticks_by_GICS:
            ####Use the sandp dataframe above to determine the shapes of the dataframes for each sector
            ##Using closing prices, as is custom
            for tick_no, tick in enumerate(ticks_by_GICS[sector]):
                d_with_ticker_dfs[sector][tick] = yf.Ticker(ticks_by_GICS[sector][tick_no]).history(period = time_period)['Close']

        ### at least 80% non NaN values
        threshold = int(round(d_with_ticker_dfs[sector].shape[0] * 0.8))
        d_with_ticker_dfs[sector].dropna(axis = 1, thresh = threshold, inplace = True)
        d_with_ticker_dfs[sector].to_csv('C:/Git stuff/Pairs_Trading/DatabySector_{}.csv'.format(sector))
        return d_with_ticker_dfs
    else:
        return None

def read_from_excel(ticks_by_GICS, excel_option):
    if excel_option:
        for sector in ticks_by_GICS:
            d_with_ticker_dfs[sector] = pd.read_csv('C:/Git stuff/Pairs_Trading/DatabySector_{}.csv'.format(sector), index_col = 0)
        return d_with_ticker_dfs
    else:
        return None

##gathers excel or yahoo data
def which_function_to_run(*args, **kwargs):
    l = []
    for func in args:
        l.append(func(*kwargs[func.__name__]))
    return l
 
d_with_ticker_dfs = [func for func in which_function_to_run(read_from_yahoo, read_from_excel, read_from_yahoo = (ticks_by_GICS, yahoo_option), read_from_excel = (ticks_by_GICS, excel_option)) if func is not None][0]

####Determining the four pairs from each sector with the highest correlations
corr_mat = {}
correlated_pairs = {} ###best pairs for each tick in each sector
for sector in ticks_by_GICS:
    corr_mat[sector] = d_with_ticker_dfs[sector].corr()
    corr_mat[sector].dropna(axis = 0, thresh = 2, inplace = True)
    corr_mat[sector].dropna(axis = 1, thresh = 2, inplace = True)
    correlated_pairs[sector] = {}
    for tick in corr_mat[sector].columns:
        ###ranking correlations 
        corr_mat[sector].sort_values(by = tick, ascending = False, inplace = True)
        
        ###as iloc[0] would give 1 iloc[1] is used
        correlation = corr_mat[sector][tick].iloc[1]
        second_stock = corr_mat[sector][tick][corr_mat[sector][tick] == correlation].index.tolist()
        correlated_pairs[sector][tick] = [correlation , second_stock[0]]
        
best_correlated_pairs_sector = {} ###best pairs overall
for sector in correlated_pairs: 
    best_correlated_pairs_sector[sector] = []
    keys_with_highest_values = sorted(correlated_pairs[sector], key=correlated_pairs[sector].get, reverse=True)[:no_pairs]
    for i in keys_with_highest_values:
        key_pair =  correlated_pairs[sector][i][1]
        if [key_pair, i] not in best_correlated_pairs_sector[sector]:
            best_correlated_pairs_sector[sector].append([i,key_pair])

####Check if the pairs pass coinegration test
#eigen_vec = []
final_pairs = [] ####Contains the pairs that pass all the tests
for sector in ticks_by_GICS:
    for i in range(len(best_correlated_pairs_sector[sector])):
        ##check for stationarity first (both stocks should be non stationary)
        d_with_ticker_dfs[sector][best_correlated_pairs_sector[sector][i][0]].dropna(inplace=True)
        d_with_ticker_dfs[sector][best_correlated_pairs_sector[sector][i][1]].dropna(inplace=True)
        if ADF(d_with_ticker_dfs[sector][best_correlated_pairs_sector[sector][i][0]]) and ADF(d_with_ticker_dfs[sector][best_correlated_pairs_sector[sector][i][1]]):
            joint_pairs_series = pd.merge(d_with_ticker_dfs[sector][best_correlated_pairs_sector[sector][i][0]], d_with_ticker_dfs[sector][best_correlated_pairs_sector[sector][i][1]], left_index=True, right_index=True)
            if get_johansen(joint_pairs_series, 0):
                if get_johansen(joint_pairs_series, 0)[0] != 0:
                    final_pairs.append((best_correlated_pairs_sector[sector][i][0], best_correlated_pairs_sector[sector][i][1]))                       

#######When to get into the pairs
####determining s.d.s of the pair-ratios
####threshold is 2 std's
df_with_just_ticks_values = d_with_ticker_dfs['Industrials'].copy() ###df with all the ticks and no sector segregation

for sector in d_with_ticker_dfs:
    if sector != 'Industrials':
        df_with_just_ticks_values = pd.merge(df_with_just_ticks_values, d_with_ticker_dfs[sector], left_index=True, right_index=True)

training_set_size = round(int(0.6 * df_with_just_ticks_values.shape[0])) ####training set (60%)
test_set_end = round(int(0.8 * df_with_just_ticks_values.shape[0])) ####test set (20%)

std_train = {}
avg_train = {}
price_ratio_training = pd.DataFrame()
std_test = {}
avg_test = {}
price_ratio_test = pd.DataFrame()
ret = []
for pair in final_pairs:
    num = df_with_just_ticks_values[pair[0]].iloc[0:training_set_size].copy()
    num.dropna(inplace = True)
    denom = df_with_just_ticks_values[pair[1]].iloc[0:training_set_size].copy()
    denom.dropna(inplace = True)
    if num.shape[0] < 0.8 * df_with_just_ticks_values[pair[0]] or denom.shape[0] < 0.8 * df_with_just_ticks_values[pair[0]]:
        continue
    
    price_ratio_training[pair[0] + pair[1]] = num.copy()/denom.copy()
    std_train[(pair[0],pair[1])] = price_ratio_training[pair[0]+pair[1]].std()
    avg_train[(pair[0],pair[1])] = price_ratio_training[pair[0]+pair[1]].mean()
    
    num_test = df_with_just_ticks_values[pair[0]].iloc[training_set_size:test_set_end].copy()
    denom_test = df_with_just_ticks_values[pair[1]].iloc[training_set_size:test_set_end].copy()
    price_ratio_test[pair[0] + pair[1]] = num_test.copy()/denom_test.copy()
    avg_test[(pair[0],pair[1])] = price_ratio_test[pair[0]+pair[1]].mean()
    
    ####if over 10 days the std is greater than the long run avg std * 2
    for i in range(0, test_set_end - training_set_size, observe_pair_period):
        returns = 0
        if std_train[(pair[0],pair[1])] * 2 < price_ratio_test[pair[0] + pair[1]].iloc[i:i+observe_pair_period].std():
            ###go long the stock that has gone down and short the one that is up trending
            if avg_train[(pair[0],pair[1])] < avg_test[(pair[0],pair[1])]:
                ###uptredning
                ###go long stock in numerator for 1-month period
                ##calculate returns
                returns += df_with_just_ticks_values[pair[0]].iloc[i+observe_pair_period:i+total_period]/ df_with_just_ticks_values[pair[0]].iloc[i+observe_pair_period:i+total_period].shift(1) - 1 \
                        - df_with_just_ticks_values[pair[1]].iloc[i+observe_pair_period:i+total_period] / df_with_just_ticks_values[pair[1]].iloc[i+observe_pair_period:i+total_period].shift(1) - 1
            else:
                ###short numerator and log denom
                returns += - df_with_just_ticks_values[pair[0]].iloc[i+observe_pair_period:i+total_period]/ df_with_just_ticks_values[pair[0]].iloc[i+observe_pair_period:i+total_period].shift(1) - 1 \
                        + df_with_just_ticks_values[pair[1]].iloc[i+observe_pair_period:i+total_period] / df_with_just_ticks_values[pair[1]].iloc[i+observe_pair_period:i+total_period].shift(1) - 1
        
        ret.append(returns)
            
##Sharpe Ratio (without costs)
rf = 1.0
sr = (ret - rf)/statistics.stdev(ret)
print(sr)
