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
import statsmodels.api as sm
import statistics
import math

time_period = '240mo' ###20 years of tick data
no_pairs = 4 ###4 pairs for each sector

costs = 5##5 percent
###need to optimize these parameters
observe_pair_period = 500 ###no of days pair is observed (calc mean and std during this period)
lookback_period = 10 ###check the pair mean and std and compare against mean and std from oberve_pair_period
trading_period = 45
threshold_in = 2
threshold_out = 0.5
rf = 1.0
transaction_costs = 4.5
stop_loss_return = -4.0

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
        
###if the spread over the observation period is past the thresholds then we enter the pair
def spreads(pairs, data_with_pair_values):
    
    Y = data_with_pair_values[pairs[0]]
    X = data_with_pair_values[pairs[1]]
    model = sm.OLS(Y,X).fit()
    
    spread = {}
    
    spread['hedge_ratio'] = model.params[0]
    
    # Create the spread and z-score of the spread
    spread['spread'] = model.resid
    spread['zscore'] = (spread['spread'] - np.mean(spread['spread']))/np.std(spread['spread'])

    return spread
        
####################################
std_train = {}
avg_train = {}
pairs_df_training = pd.DataFrame()
std_test = {}
avg_test = {}
price_ratio_test = pd.DataFrame()
ret = []
    
for pair in final_pairs:
    #num = df_with_just_ticks_values[pair[0]].iloc[0:training_set_size].copy()
    first_stock = df_with_just_ticks_values[pair[0]].copy()
    #denom = df_with_just_ticks_values[pair[1]].iloc[0:training_set_size].copy()
    second_stock = df_with_just_ticks_values[pair[1]].copy()
    pairs_df = pd.merge(first_stock, second_stock, left_index= True, right_index = True)
    pairs_df.dropna(inplace=True)
    
    training_set_size = round(int(0.6 * pairs_df.shape[0]))
    test_set_end = round(int(0.8 * pairs_df.shape[0]))
    
    if training_set_size < 0.5 * round(int(0.6 * df_with_just_ticks_values.shape[0])):
        continue
    
    first_stock_test = df_with_just_ticks_values[pair[0]].iloc[training_set_size:test_set_end].copy()
    second_stock_test = df_with_just_ticks_values[pair[1]].iloc[training_set_size:test_set_end].copy()

    ###Check the mean Z-Score between the lookback period in comparison to the observation period
    ###to see if threshold_in gets breached. 
    ###And then keep checking trading period Z-Score to ensure the trades havent breached the threshold_out 
    for i in range(observe_pair_period, training_set_size - trading_period - 1, lookback_period):
            ###std up to lookback period is less than std during lookback period
            returns = 0
            flag = 0
            spread_obs_period = spreads(pair, pairs_df.iloc[0:i])
            spread_lookback_period = spreads(pair, pairs_df.iloc[i:i+lookback_period])
            z_score_obs_period = np.mean(spread_obs_period['zscore'])
            z_score_lookback_period = np.mean(spread_lookback_period['zscore'])
            if z_score_obs_period * threshold_in < z_score_lookback_period:
                ###Pairs have divereged
                ###Long 2nd and short 1st
                for t in range(2, trading_period):
                    flag = 1
                    spread_trading_period = spreads(pair, pairs_df.iloc[i+lookback_period:i+lookback_period+t])
                    z_score_trading_period = np.mean(spread_trading_period['zscore'])
                    if z_score_trading_period > threshold_out * z_score_obs_period: 
                        if returns > stop_loss_return:
                            returns += -1 * (spread_trading_period['hedge_ratio'] * df_with_just_ticks_values[pair[0]].iloc[i+lookback_period+t]/ df_with_just_ticks_values[pair[0]].iloc[i+lookback_period+t-1] - 1) \
                            + 1 * (df_with_just_ticks_values[pair[1]].iloc[i+lookback_period+t] / df_with_just_ticks_values[pair[1]].iloc[i+lookback_period+t-1] - 1)
                        else: 
                            ## if the loss is more than the stop loss
                            break
                    else:
                        ## if the  pairs have converged to a large extent
                            break
            elif z_score_obs_period > threshold_in * z_score_lookback_period:
                ###Long 1st and short 2nd
                for t in range(2, trading_period):
                    flag = 2
                    spread_trading_period = spreads(pair, pairs_df.iloc[i+lookback_period:i+lookback_period+t])
                    z_score_trading_period = np.mean(spread_trading_period['zscore'])
                    if z_score_trading_period * threshold_out < z_score_obs_period: 
                        if returns > stop_loss_return:
                            returns += 1  * (df_with_just_ticks_values[pair[0]].iloc[i+lookback_period+t]/ df_with_just_ticks_values[pair[0]].iloc[i+lookback_period+t-1] - 1) \
                            - 1 * spread_trading_period['hedge_ratio'] * (df_with_just_ticks_values[pair[1]].iloc[i+lookback_period+t] / df_with_just_ticks_values[pair[1]].iloc[i+lookback_period+t-1] - 1)
                        else:
                            break
                    else:
                            break

            if returns!=0:
                if not math.isnan(returns):
                    ret.append(returns*100) 
    
##Sharpe Ratio (without costs)
sr = (statistics.mean(ret) - rf - transaction_costs)/statistics.stdev(ret)
    
print(sr)
