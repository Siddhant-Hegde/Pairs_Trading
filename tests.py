# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:52:37 2020

@author: ss466
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR

def ADF(v, crit='5%', max_d=6, reg='nc', autolag='AIC'):
    """ Augmented Dickey Fuller test

    Parameters
    ----------
    v: ndarray matrix
        residuals matrix

    Returns
    -------
    bool: boolean
        true if v pass the test 
    """

    boolean = False

    
    adf = adfuller(v, max_d, reg, autolag)
    if(adf[0] < adf[4][crit]):
        boolean = False ###Reject the NULL that there is a Unit Root
    else:
        boolean = True ####Cant reject NULL

    return boolean

def get_johansen(y, p):
        """
        Get the cointegration vectors at 95% level of significance
        given by the trace statistic test.
        """
        return_vec = []
        try:
            result = coint_johansen(y, det_order=0, k_ar_diff=p)
            result_table = np.hstack((np.expand_dims(np.round(result.lr2,4),axis=1),result.cvm))
            result_evec = np.round(result.evec,4)
            #print('This is the result table {}'.format(result_table))
            for i in range(result_table.shape[0]):
                if result_table[i][0] > result_table[i][2]:
                    continue
                else:
                    return_vec.append(i)
                    break
#            if return_vec is not None:
#                highest_eigval_indx = np.argmax(np.max(result.eig, axis=0))
#                highest_eigvec = result_evec[:,highest_eigval_indx]
#                return_vec.append(list(highest_eigvec))
            return return_vec
        except np.linalg.LinAlgError:
            return None
            
            