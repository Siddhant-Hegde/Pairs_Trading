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
from functools import reduce
from itertools import chain

time_period = '240mo' ###20 years of tick data
no_pairs = 4 ###4 pairs for each sector

###need to optimize these parameters
formation_period = 180
observe_pair_period = 300 ###no of days pair is observed (calc mean and std during this period)
lookback_period = 2 ###check the pair mean and std and compare against mean and std from oberve_pair_period
trading_period = 30
threshold_in = 2
threshold_out = 0.5
rf = 0.5
stop_loss_return = -4.0

####Do we want to scrape the data from yahoo or used excel files that contain data from when the yahoo data 
####was previously scraped
yahoo_option = 0 ##get yahoo data
excel_option = 1 - yahoo_option

sandp = yf.Ticker('^GSPC').history(time_period)

######splitting by sector
GICS_df = pd.read_csv('C:/Git stuff/Pairs_Trading/GICS split.csv', encoding= 'unicode_escape')

###read the data from yahoo
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

###read the data from excel 
###this data was written to excel from yahoo
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
 

####Determining the four pairs from each sector with the highest correlations
def highest_correlated(ticks_by_GICS, d_with_ticker_dfs, no_pairs, strategy_run_day, formation_period):
    
    ###first determine the different correlations
    corr_mat = {}
    correlated_pairs = {} ###best pairs for each tick in each sector
    for sector in ticks_by_GICS:
        corr_mat[sector] = d_with_ticker_dfs[sector].iloc[strategy_run_day:strategy_run_day+formation_period].corr()
        corr_mat[sector].dropna(axis = 0, thresh = 2, inplace = True)
        corr_mat[sector].dropna(axis = 1, thresh = 2, inplace = True)
        correlated_pairs[sector] = {}
        for tick in corr_mat[sector].columns:
            ###ranking correlations 
            corr_mat[sector].sort_values(by = tick, ascending = False, inplace = True)
            
            ###as iloc[0] would give 1 iloc[1] is used
            correlation = corr_mat[sector][tick].iloc[1]
            second_stock = corr_mat[sector][tick][corr_mat[sector][tick] == correlation].index.tolist()
            correlated_pairs[sector][tick] = [correlation , second_stock[0]] ###the correlations
    
    ###determine the pairs with the highest correlations 
    best_correlated_pairs_sector = {} ###best pairs overall
    for sector in correlated_pairs: 
        best_correlated_pairs_sector[sector] = []
        keys_with_highest_values = sorted(correlated_pairs[sector], key=correlated_pairs[sector].get, reverse=True)[:no_pairs]
        for i in keys_with_highest_values:
            key_pair =  correlated_pairs[sector][i][1]
            if [key_pair, i] not in best_correlated_pairs_sector[sector]:
                best_correlated_pairs_sector[sector].append([i,key_pair])
        
    return best_correlated_pairs_sector

####Check if the pairs pass coinegration test
def obtain_final_pairs(ticks_by_GICS,d_with_ticker_dfs, no_pairs, strategy_run_day, formation_period):
    #eigen_vec = []
    final_pairs = [] ####Contains the pairs that pass all the tests
    best_correlated_pairs_sector = highest_correlated(ticks_by_GICS, d_with_ticker_dfs, no_pairs, strategy_run_day, formation_period)
    
    for sector in ticks_by_GICS:
        for i in range(len(best_correlated_pairs_sector[sector])):
            ##check for stationarity first (both stocks should be non stationary)
            stock_1 = d_with_ticker_dfs[sector][best_correlated_pairs_sector[sector][i][0]].iloc[strategy_run_day:strategy_run_day+formation_period].dropna()
            stock_2 = d_with_ticker_dfs[sector][best_correlated_pairs_sector[sector][i][1]].iloc[strategy_run_day:strategy_run_day+formation_period].dropna()
            if ADF(stock_1) and ADF(stock_2):
                joint_pairs_series = pd.merge(stock_1, stock_2, left_index=True, right_index=True)
                if get_johansen(joint_pairs_series, 0):
                    if get_johansen(joint_pairs_series, 0)[0] != 0:
                        final_pairs.append((best_correlated_pairs_sector[sector][i][0], best_correlated_pairs_sector[sector][i][1]))                       

    return final_pairs

###if the spread over the observation period is bigger than the thresholds then we enter the pair
def spreads(pairs, data_with_pair_values):
    
    Y = data_with_pair_values[pairs[0]]
    X = data_with_pair_values[pairs[1]]
    model = sm.OLS(Y,X).fit()
    
    spread = {}
    
    spread['hedge_ratio'] = model.params[0]
    
    # Calculate the spread and z-score of the spread
    spread['spread'] = model.resid
    spread['zscore'] = (spread['spread'] - np.mean(spread['spread']))/np.std(spread['spread'])

    return spread

###calculating the returns from the strategy
def returns_from_strategy(final_pairs, d_with_ticker_dfs, df_with_just_ticks_values, strategy_run_day, df_with_pairs, formation_period, observe_pair_period):
    
    ret = []
    
    for pair in final_pairs:
        
        pairs_df = df_with_pairs[[pair[0],pair[1]]]
        ###Check the mean Z-Score between the lookback period in comparison to the observation period
        ###to see if threshold_in gets breached. 
        ###And then keep checking trading period Z-Score to ensure the trades havent breached the threshold_out 
            ###std up to lookback period is less than std during lookback period
        daily_returns = []
        cumulative_returns = 0
        flag = 0
        spread_obs_period = spreads(pair, pairs_df.iloc[strategy_run_day:strategy_run_day+observe_pair_period])
        spread_lookback_period = spreads(pair, pairs_df.iloc[strategy_run_day+observe_pair_period:strategy_run_day+observe_pair_period+lookback_period])
        z_score_obs_period = np.mean(spread_obs_period['zscore'])
        z_score_lookback_period = np.mean(spread_lookback_period['zscore'])
        if z_score_obs_period * threshold_in < z_score_lookback_period and flag == 0:
            ###Pairs have divereged
            ###Long 2nd and short 1st
            for t in range(2, trading_period):
                
                spread_trading_period = spreads(pair, pairs_df.iloc[strategy_run_day+observe_pair_period+lookback_period:strategy_run_day+observe_pair_period+lookback_period+t])
                z_score_trading_period = np.mean(spread_trading_period['zscore'])
                
                if z_score_trading_period > threshold_out * z_score_obs_period and cumulative_returns > stop_loss_return: 
                        
                        num = df_with_just_ticks_values[pair[0]].iloc[strategy_run_day+observe_pair_period+lookback_period+t]
                        denom = df_with_just_ticks_values[pair[0]].iloc[strategy_run_day+observe_pair_period+lookback_period+t-1]
                        num_1 = df_with_just_ticks_values[pair[1]].iloc[strategy_run_day+observe_pair_period+lookback_period+t]
                        denom_1 = df_with_just_ticks_values[pair[1]].iloc[strategy_run_day+observe_pair_period+lookback_period+t-1]
                        
                        daily_returns.append((-1 * (spread_trading_period['hedge_ratio'] * (num/denom - 1)) + 1 * ((num_1/denom_1)-1)) + 1)
                        
                        cumulative_returns = 100*((reduce(lambda x, y: x*y,daily_returns))-1)
                        
                else:
                    ## if the  pairs have converged to a large extent
                        break
        elif z_score_obs_period > threshold_in * z_score_lookback_period and flag == 0:
            ###Long 1st and short 2nd
            for t in range(2, trading_period):
                
                spread_trading_period = spreads(pair, pairs_df.iloc[strategy_run_day+observe_pair_period+lookback_period:strategy_run_day+observe_pair_period+lookback_period+t])
                z_score_trading_period = np.mean(spread_trading_period['zscore'])
                
                if z_score_trading_period * threshold_out < z_score_obs_period and cumulative_returns > stop_loss_return: 
                        
                        num = df_with_just_ticks_values[pair[0]].iloc[strategy_run_day+observe_pair_period+lookback_period+t]
                        denom = df_with_just_ticks_values[pair[0]].iloc[strategy_run_day+observe_pair_period+lookback_period+t-1]
                        num_1 = df_with_just_ticks_values[pair[1]].iloc[strategy_run_day+observe_pair_period+lookback_period+t]
                        denom_1 = df_with_just_ticks_values[pair[1]].iloc[strategy_run_day+observe_pair_period+lookback_period+t-1]
                        daily_returns.append((1 * ((num/denom - 1)) - 1 * (spread_trading_period['hedge_ratio'] * (num_1/denom_1-1))) + 1)

                        cumulative_returns = 100*((reduce(lambda x, y: x*y,daily_returns))-1)
                        
                else:
                    
                        break

        if cumulative_returns!=0:
            if not math.isnan(cumulative_returns):
                ret.append(cumulative_returns) 
    
    return ret

###running the strategy###
##########################    

###dataframe of sector: list of tickers
ticks_by_GICS = {sector: list(GICS_df['Symbol'][GICS_df[GICS_df.columns[3]]==sector]) for sector in GICS_df[GICS_df.columns[3]].unique()}

###dataframe just containing tick data (no sector segmentation)
d_with_ticker_dfs = {sector: pd.DataFrame(index = sandp.index, columns = ticks_by_GICS[sector], data = np.zeros((sandp.shape[0] ,len(ticks_by_GICS[sector])))) for sector in ticks_by_GICS}
d_with_ticker_dfs = [func for func in which_function_to_run(read_from_yahoo, read_from_excel, read_from_yahoo = (ticks_by_GICS, yahoo_option), read_from_excel = (ticks_by_GICS, excel_option)) if func is not None][0]

df_with_just_ticks_values = d_with_ticker_dfs['Industrials'].copy() ###df with all the ticks and no sector segregation

for sector in d_with_ticker_dfs:
    if sector != 'Industrials':
        df_with_just_ticks_values = pd.merge(df_with_just_ticks_values, d_with_ticker_dfs[sector], left_index=True, right_index=True)
    
###obtaining the pairs
strategy_run_day = 0

final_pairs = obtain_final_pairs(ticks_by_GICS,d_with_ticker_dfs, no_pairs, strategy_run_day, formation_period)
flat_list_pairs = list(set(chain(*final_pairs)))

pairs_df = df_with_just_ticks_values[flat_list_pairs].copy()
pairs_df.dropna(inplace=True)
        
training_set_size = round(int(pairs_df.shape[0]))

returns = []
while strategy_run_day + formation_period + observe_pair_period + trading_period + lookback_period < training_set_size:
    
    ret = returns_from_strategy(final_pairs, d_with_ticker_dfs, df_with_just_ticks_values, strategy_run_day, pairs_df, formation_period, observe_pair_period)
    returns.append(ret)
    strategy_run_day = strategy_run_day + formation_period + observe_pair_period + trading_period + lookback_period
    final_pairs = obtain_final_pairs(ticks_by_GICS, d_with_ticker_dfs, no_pairs, strategy_run_day, formation_period)
    final_pairs_flattened = list(set(chain(*final_pairs)))
    pairs_df = df_with_just_ticks_values[final_pairs_flattened].copy()


##Sharpe Ratio
flat_ret = [item for sublist in returns for item in sublist]

sr = (statistics.mean(flat_ret) - rf)/statistics.stdev(flat_ret)
    
print(sr)
