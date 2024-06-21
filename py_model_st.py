import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

from typing import List

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

import pmdarima as pm

from abc import ABC, abstractmethod


def lc_svd_est(lmxt):
    """
    Plain Lee-Carter model
    estimates log(q_{x,t}) = a_x + b_x * \kappa_t
    Under Gaussian innovations, estimate via Singular Value Decomposition
    x=1,...,N (ages), t=1,...T (years)
    constaints_ \sum_x bx=1, \sum_t \kappa_t=0
    Args:
        mxt: Numpy ndarray (NxT) of log mortality rates m_{x,t}, x=age, t=year
    Returns:
        A list of vectors of estimated coefficients along with the forecasted mortality in the following order:
            - ax_hat Nx1,
            - bx_hat Nx1,
            - kt_hat 1xt
            - mxt_hat the forecasted mortality
    """
    N, T = lmxt.shape

    # average age effect
    ax_hat  = np.mean(lmxt,1) # averaging for each age (along the second dimension)

    # substract the average age effect from the mortality rates (using broadcasting concept)
    z_xt = lmxt - ax_hat.reshape(N, 1)

    # SVD
    # U python = U in R (except for column > 62 from 64 columns)
    # V python = V.T in R
    # S python = S in R
    U, S, V = np.linalg.svd(z_xt, full_matrices=True)

    # get the estimation for bx and kt
    bx_hat = -U[:, 0] * S[0]
    kt_hat = -V.T[:, 0]

    # standardization procedure
    # sum_x bx=1, sum_t kt=0
    sum_b  = np.sum(bx_hat)
    bx_hat = bx_hat/sum_b
    kt_hat = kt_hat*sum_b
    sum_k  = np.mean(kt_hat) # this is a negligible number 10^-14
    kt_hat = kt_hat-sum_k
    ax_hat = ax_hat + bx_hat*sum_k

    # reconstruction of parameters to mortality rates
    # outer product between bx and kt, then add ax to all ages (rows) for each year t (column)
    mxt_hat = np.exp(ax_hat.reshape(-1,1) + np.einsum('i,j->ij', bx_hat, kt_hat))

    return(ax_hat, bx_hat, kt_hat, mxt_hat)

def get_ext_dxt(qxt):
    """
    Estimate the number of death for each age/year as well as the number of
    exposures (that is the population at beginning of year) based on the given matrix of probability of death.
    Start with cohorts of 100'000 in the first year.
    Args:
        mxt: Matrix of probability of death, with ages for rows and year for columns.
    Returns:
        Two matrix (of size ages x years) for the exposure and the number of deaths for each age and year.
    """
    num_ages, num_years = qxt.shape

    # Initialize Ext with zeros, then set the first row to 100,000 (age 0 for each year)
    Ext = np.zeros((num_ages+1, num_years))
    Ext[0,:] = 100000

    # Calculate the Ext values
    for t in range(1, num_ages + 1):
        Ext[t, :] = Ext[t - 1, :] * (1 - qxt[t - 1, :])

    # Calculate the number of deaths
    Dxt = Ext[:-1, :] - Ext[1:, :]

    # Update Ext to remove the last row
    Ext = Ext[:-1, :]

    # Return
    return(Ext, Dxt)

def lc_poisson_step(params_est:List[np.ndarray[np.float32]], Dxt, Ext):
    """
    The argument params_est contains the estimated parameters of the LC model that would be updated in this step.
    It is a list of 3 numpy arrays, which is the ax, bx, and kt in that order.
    The three estimates will be updated via iteration likelihood set for Poisson density.
    Args:
        params_est (List[np.ndarray[np.float32]]): a list of three np array containing initial estimates of ax, bx, and kt respectively
        Dxt: a matrix of the number of death in a particular age (row) and year (column)
        Ext: a matrix of the exposure (exposure-to-risk) in a particular age (row) and year (column)
    Returns:
        A list of 3 numpy arrays similar to params_est, containing the updated estimates of the three parameters.
    """
    ax = params_est[0]
    bx = params_est[1]
    kt = params_est[2]

    # First, update the estimates for ax
    # Compute mxt using element-wise operations (this is equivalent to what is used in lc_svd to reconstruct the mortality matrix from the estimates)
    mxt = np.exp(ax[:, np.newaxis] + bx[:, np.newaxis] * kt[np.newaxis, :])
    Dxhat0 = Ext * mxt
    ax_new = ax + np.sum(Dxt - Dxhat0, axis=1) / np.sum(Dxhat0, axis=1)

    # Second, update the estimates for kt
    mxt = np.exp(ax_new[:, np.newaxis] + bx[:, np.newaxis] * kt[np.newaxis, :])
    Dxhat1 = Ext * mxt
    # kt_new = kt + np.sum((Dxt - Dxhat1) * bx[:, np.newaxis], axis=0) / np.sum(Dxhat1 * (bx[:, np.newaxis] ** 2), axis=0)
    # CHECK THE FORMULA WITH PROF
    kt_new = kt + np.sum((Dxt - Dxhat1) * bx[:, np.newaxis], axis=0) / np.sum(Dxhat0 * (bx[:, np.newaxis] ** 2), axis=0)

    # Finally, update the estimates for bx
    mxt = np.exp(ax_new[:, np.newaxis] + bx[:, np.newaxis] * kt_new[np.newaxis, :])
    Dxhat2 = Ext * mxt
    bx_new = bx + np.sum((Dxt - Dxhat2) * kt_new[np.newaxis, :], axis=1) / np.sum(Dxhat2 * (kt_new[np.newaxis, :] ** 2), axis=1)

    # Standardization procedure
    # CHECK THE FORMULA WITH PROF
    # mean_k = np.mean(kt_new)
    sum_b = np.sum(bx_new)
    bx_new = bx_new / sum_b
    kt_new = (kt_new - np.mean(kt_new)) * sum_b
    mean_k = np.mean(kt_new)
    kt_new = kt_new - mean_k
    ax_new = ax_new + bx_new * mean_k

    return [ax_new, bx_new, kt_new]

def lc_poisson_est(mxt, Ext, Dxt=None):
    """
    Estimates the parameters of the Lee-Carter model using the Poisson likelihood assumption and
    Newton-Rhapson iterative method.
    Args:
        mxt - a matrix of mortality rates with as many rows as there are ages and as many columns as there are years
        Ext - a matrix of exposures with as many rows as there are ages and as many columns as there are years
        Dxt - a matrix of the number of deaths with as many rows as there are ages and as many columns as there are years
    Returns:
        A list of vectors of estimated coefficients along with the forecasted mortality in the following order:
        - ax_hat Nx1,
        - bx_hat Nx1,
        - kt_hat 1xt
        - mxt_hat the forecasted mortality
    """
    # Check whether the necessary inputs are provided 
    if((mxt is None) and (Dxt is None)):
        raise ValueError("One of mxt (mortality rates) or Dxt (number of deaths) needs to be specified")
    elif(mxt is None):
        mxt = Dxt/Ext
    else:
        Dxt = mxt*Ext

    dist=10**10
    iterCtr=0
    (ax_prev, bx_prev, kt_prev, _) = lc_svd_est(np.log(mxt))
    while(dist > 10**(-10) and iterCtr < 10000):
        iterCtr=iterCtr+1
        ax_curr, bx_curr, kt_curr = lc_poisson_step([ax_prev, bx_prev, kt_prev], Dxt, Ext)
        # as first described in Brouhns, Denuit, Vermunt
        dist = (sum(abs(ax_curr-ax_prev)) +
                sum(abs(bx_curr-bx_prev)) +
                sum(abs(kt_curr-kt_prev)))
        ax_prev = ax_curr.copy()
        bx_prev = bx_curr.copy()
        kt_prev = kt_curr.copy()
    
    # print("Iteration: ", iterCtr)
    # print("Distance: ", dist, dist < 10**(-10))        
    
    # reconstruction of parameters to mortality rates
    # outer product between bx and kt, then add ax to all ages (rows) for each year t (column)
    mxt_hat = np.exp(ax_curr.reshape(-1,1) + np.einsum('i,j->ij', bx_curr, kt_curr))

    return [ax_curr, bx_curr, kt_curr, mxt_hat]

def get_mxt_cohort(ax,bx_1,kt,bx_2,gz,
                     Ns1:int, Ns2:int):
    """
    computes the expected mortality rates based on the given parameters from a Renshaw-Haberman cohort model (ax, bx_1, kt, bx_2, gz).
    Args:        
        ax: the average mortality rates for each age x
        bx_1: the sensitivity of each age x to the first time parameter
        kt: the time parameter 
        bx_2: the sensitivity of each age x to the cohort parameter
        gz: the cohort parameter
        Ns1 (int): the first few cohorts to be excluded
        Ns2 (int): the last few cohorts to be excluded
    """
    N = len(ax)
    T = len(kt)
    Nc = len(gz)
    eh = np.ones(T)
    ev = np.ones(N)
    
    # Get the expected mortality (mxt) from the first three parameters (standard LC)
    X1 = np.outer(ax, eh) + np.outer(bx_1, eh) * kt
    
    # Get the expected mortality (mxt) from the cohort effect
    X2 = np.zeros((N, T))    
    for idx_coh in range(Nc):
        # if idx_coh <= N - Ns1:
            # z = N - Ns1 - idx_coh
        if idx_coh < N - Ns1:
            z = N - Ns1 - (idx_coh+1)
            for ctr in range(min(T, N - z)):
                # X2[z + ctr, ctr] = idx_coh
                X2[z + ctr, ctr] = bx_2[z + ctr] * gz[idx_coh]
        else:
            z = (idx_coh+1) - (N - Ns2) + 1
            for ctr in range(min(N, T - z + 1)):
                # X2[ctr, z + ctr - 1] = idx_coh
                X2[ctr, z + ctr - 1] = bx_2[ctr] * gz[idx_coh]
    
    return np.exp(X1 + X2)
    
def get_mxt_dxt_diff(Dxt,Ext,ax,bx_1,kt,bx_2,gz,
                     Ns1:int, Ns2:int):
    """
    computes computes the expected mortality rates, the expected number of death, and 
    the difference between the expected and the real number of death
    based on the given parameters from a Renshaw-Haberman cohort model (ax, bx_1, kt, bx_2, gz).

    computes expected number of death from given parameters, 
    as well as difference between the expected and the observed Dxt

    Args:
        Dxt:
        Ext:
        ax: the average mortality rates for each age x
        bx_1: the sensitivity of each age x to the first time parameter
        kt: the time parameter 
        bx_2: the sensitivity of each age x to the cohort parameter
        gz: the cohort parameter
        Ns1 (int): the first few cohorts to be excluded
        Ns2 (int): the last few cohorts to be excluded
    Return:
        mxt_hat
        Dxt_hat
        diff_Dxt
    """    
    mxt_hat = get_mxt_cohort(ax, bx_1, kt, bx_2,gz, Ns1, Ns2)
    Dxt_hat = mxt_hat * Ext
    diff_Dxt = Dxt - Dxt_hat  # corresponds to y - y_hat
    
    return mxt_hat, Dxt_hat, diff_Dxt

def rh_poisson_step(params_est:List[np.ndarray[np.float32]], Dxt, Ext,
                    Ns1:int, Ns2:int):
    """
    The argument params_est contains the estimated parameters of the RH model that would be updated in this step.
    It is a list of 5 numpy arrays, which is the ax, bx_1, kt, bx_2, gz in that order.
    The five estimates will be updated via iteration likelihood set for Poisson density.
    Args:
        params_est (List[np.ndarray[np.float32]]): a list of five np array containing initial estimates of ax, bx_1, kt, bx_2, gz respectively
        Dxt: a matrix of the number of death in a particular age (row) and year (column)
        Ext: a matrix of the exposure (exposure-to-risk) in a particular age (row) and year (column)
        Ns1: the first few cohorts to be excluded
        Ns2: the last few cohorts to be excluded
    Returns:
        A list of 3 numpy arrays similar to params_est, containing the updated estimates of the three parameters.
    """    
    ax = params_est[0]
    bx_1 = params_est[1]
    kt = params_est[2]
    bx_2 = params_est[3]
    gz = params_est[4]

    # Get the total number of ages and years estimated
    N = len(ax)
    T = len(kt)
    Nc = len(gz)

    #######################################################################################################
    # First Update ax
    # Get estimate of the number of deaths (and the diff with the real Dxt)    
    _, Dxt_hat, diff_Dxt = get_mxt_dxt_diff(Dxt, Ext, ax, bx_1, kt, bx_2, gz, Ns1, Ns2)
    ax_new = ax + np.sum(diff_Dxt, axis=1) / np.sum(Dxt_hat, axis=1)

    #######################################################################################################
    # Update kt
    _, Dxt_hat, diff_Dxt = get_mxt_dxt_diff(Dxt, Ext, ax_new, bx_1, kt, bx_2, gz, Ns1, Ns2)
    kt_new = kt + np.sum(diff_Dxt * bx_1[:, None], axis=0) / np.sum(Dxt_hat * (bx_1[:, None] ** 2), axis=0)

    #######################################################################################################
    # Update bx_1
    _, Dxt_hat, diff_Dxt = get_mxt_dxt_diff(Dxt, Ext, ax_new, bx_1, kt_new, bx_2, gz, Ns1, Ns2)
    bx_1_new = bx_1 + np.sum(diff_Dxt * kt_new, axis=1) / np.sum(Dxt_hat * (kt_new ** 2), axis=1)
    
    # Rescale bx_1 and kt (standardization in LC Poisson)
    kt_new = kt_new - np.mean(kt_new)
    c = np.sum(bx_1_new)
    bx_1_new = bx_1_new / c
    kt_new = kt_new * c
         
    #######################################################################################################
    # Update gamma_z (more complicated because of cohort indexing need to match the age-year rate matrix)
    _, Dxt_hat, diff_Dxt = get_mxt_dxt_diff(Dxt, Ext, ax_new, bx_1_new, kt_new, bx_2, gz, Ns1, Ns2)
    n_update = diff_Dxt * bx_2[:, None]
    d_update = Dxt_hat * (bx_2[:, None] ** 2)
    
    # match the indexing of the sum on the numerator and denumerator for updating gz
    sn_update = np.zeros(Nc)
    sd_update = np.zeros(Nc)    
    for idx_coh in range(Nc):
        if idx_coh < N - Ns1:
            z = N - Ns1 - (idx_coh+1)
            for ctr in range(min(T, N - z)):
                sn_update[idx_coh] += n_update[z + ctr, ctr]
                sd_update[idx_coh] += d_update[z + ctr, ctr]
        else:
            z = (idx_coh+1) - (N - Ns2) + 1
            for ctr in range(min(N, T - z + 1)):
                sn_update[idx_coh] += n_update[ctr, z + ctr - 1]
                sd_update[idx_coh] += d_update[ctr, z + ctr - 1]

    gz_new = gz + sn_update / sd_update
    gz_new = gz_new - np.mean(gz_new)

    #######################################################################################################        
    # Update bx_2    
    _, Dxt_hat, diff_Dxt = get_mxt_dxt_diff(Dxt, Ext, ax_new, bx_1_new, kt_new, bx_2, gz_new, Ns1, Ns2)    
    gz_n = np.zeros((N, T))
    gz_d = np.zeros((N, T))
    
    # the sum on the numerator and denumerator need to be constructed manually because of the gz vector
    for idx_coh in range(Nc):
        if idx_coh < N - Ns1:
            z = N - Ns1 - (idx_coh+1)
            for ctr in range(min(T, N - z)):
                gz_n[z + ctr, ctr] = gz_new[idx_coh]
                gz_d[z + ctr, ctr] = gz_new[idx_coh] ** 2
        else:
            z = (idx_coh+1) - (N - Ns2) + 1
            for ctr in range(min(N, T - z + 1)):
                gz_n[ctr, z + ctr - 1] = gz_new[idx_coh]
                gz_d[ctr, z + ctr - 1] = gz_new[idx_coh] ** 2
    
    num = np.sum(diff_Dxt * gz_n, axis=1)
    dum = np.sum(Dxt_hat * gz_d, axis=1)
    bx_2_new = bx_2 + num / dum
    
    gz_new = gz_new - np.mean(gz_new)
    c = np.sum(bx_2_new)
    bx_2_new = bx_2_new / c
    gz_new = gz_new * c
        
    return [ax_new, bx_1_new, kt_new, bx_2_new, gz_new]

def rh_poisson_est(mxt, Ext, Ns1, Ns2, Dxt=None):
    """
    Estimates the parameters of the Renshaw-Haberman mortality model (ax, bx_1, kt, bx_2, gz)
    Args:
        mxt: 
        Ext: 
        Ns1: 
        Ns2: 
        Dxt: 
    Returns:
        a list of numpy array, each contains the estimated parameters in the following order:
        - ax: the average of log mortality
        - bx_1: 
        - kt: the time parameter
        - bx_2: 
        - gz: the cohort effect
    """

    # Check whether the necessary inputs are provided 
    if(mxt is None and Dxt is None):
        raise ValueError("One of mxt (mortality rates) or Dxt (number of deaths) needs to be specified")
    elif(mxt is None):
        mxt = Dxt/Ext
    else:
        Dxt = mxt*Ext

    # Calculate the rates and the number of ages and years 
    N, T = Dxt.shape 
    Nc = N-Ns1+T-Ns2-1

    # Initial values for ax, bx_1, and kt using LC Poisson
    (ax_prev, bx_1_prev, kt_prev, _) = lc_poisson_est(mxt, Ext)

    # Initial value for the two new parameters
    gz_prev=0.1*np.ones(Nc)
    bx_2_prev = 0.01*np.ones(N)
    
    # Iteratively calculate the estimates
    dist=10**10
    iterCtr=0    
    while(dist > 0.00001 and iterCtr < 10000):
        iterCtr=iterCtr+1
        ax_curr, bx_1_curr, kt_curr, bx_2_curr, gz_curr = rh_poisson_step([ax_prev, bx_1_prev, kt_prev, bx_2_prev, gz_prev], Dxt, Ext,
                                                                          Ns1, Ns2)
        
        # as first described in Brouhns, Denuit, Vermunt
        dist = (sum(abs(ax_curr-ax_prev)) +
                sum(abs(bx_1_curr-bx_1_prev)) +
                sum(abs(kt_curr-kt_prev)) + 
                sum(abs(bx_2_curr-bx_2_prev)) + 
                sum(abs(gz_curr-gz_prev)))
        ax_prev = ax_curr.copy()
        bx_1_prev = bx_1_curr.copy()
        kt_prev = kt_curr.copy()
        bx_2_prev = bx_2_curr.copy()
        gz_prev = gz_curr.copy()
    
    # print("Iteration: ", iterCtr)
    # print("Distance: ", dist, dist < 10^(-10))
        
    # reconstruction of parameters to mortality rates
    mxt_hat, _, _ = get_mxt_dxt_diff(Dxt, Ext, ax_curr, bx_1_curr, kt_curr, bx_2_curr, gz_curr, Ns1, Ns2)

    return (ax_curr, bx_1_curr, kt_curr, bx_2_curr, gz_curr, mxt_hat)

class Lc():    
    def fit(self, mxt, ext, year_start=0, year_end=0):
        """
        Fit the model with the given data
        Args:  
            mxt: a matrix of mortality rates with ages for rows and years for columns
            year_start: (optional) the first year of the fitted data, used to construct index for plotting (if given)
            year_end: (optional) the last year of the fitted data, used to construct index for plotting (if given)
        Returns:
        """
        # Store the data to calculate residuals
        self.mxt = mxt
        self.ext = ext

        # Determine the index (year) of the data (if givem)
        self.idx_start = year_start
        self.idx_end = year_end
        if(self.idx_end == 0):
            self.idx_end = mxt.shape[1]
        
        # Estimations of Lee-Carter Parameters
        [ax_hat, bx_hat, kt_hat, _] = lc_poisson_est(mxt=self.mxt, Ext=self.ext)
        self.params = {'ax': ax_hat,
                       'bx': bx_hat, 
                       'kt': kt_hat}
    

    def predict(self, horizon=0, order=(0, 1, 0), mxt_true=None):
        """
        Predict mortality rates a few years (horizon) ahead of the fitted data. 
        The default horizon is 0, which will predict mortality rates in the same years as the fitted data.
        Args:  
            horizon: the number of future years predicted
            y_true: the true mortality rates of future years
        Returns:
            A list of numpy array containing the following information (in order):
                - The forecasted kappa_t
                - The index (or year) of the forecasted parameters
                - The predicted mortality rates for all considered ages as rows and future years as columns
                * The lower bound of the predicted mortality rates
                * The upper bound of the predicted mortality rates
                - The error (raw difference) of the predicted mortality rates (if y_true is given or horizon=0)
        """
        # If horizon = 0, predict the mortality rates of the fitted data
        if(horizon == 0):
            # reconstruction of parameters to mortality rates
            # outer product between bx and kt, then add ax to all ages (rows) for each year t (column)
            mxt_hat = np.exp(self.params['ax'].reshape(-1,1) + 
                             np.einsum('i,j->ij', self.params['bx'], self.params['kt']))
            return [self.params['kt'], np.arange(self.idx_start, self.idx_end+1), mxt_hat, self.mxt - mxt_hat]
        
        # If horizon > 0, predict the mortality rates of future years ahead of the fitted data
        else:
            # First make an ARIMA model
            if(order is None):
                model_kt = pm.auto_arima(self.params['kt'],                       
                      seasonal=False,  # TRUE if seasonal series
                      d=None,             # let model determine 'd'
                      test='adf',         # use adftest to find optimal 'd'
                      start_p=0, start_q=0, # minimum p and q
                      max_p=7, max_q=7, # maximum p and q
                      D=None,             # let model determine 'D'
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)             
                self.order = model_kt.order   
                
                # Forecast the time parameter kt according to ARIMA(order)            
                kt_forecast = model_kt.predict(n_periods=horizon)
            else:
                model_kt = ARIMA(self.params['kt'], order=order, trend='t')
                self.order = order
                model_kt = model_kt.fit()

                # Forecast the time parameter kt according to ARIMA(order)            
                kt_forecast = model_kt.forecast(steps=horizon)
            
            
            kt_all = np.append(self.params['kt'], kt_forecast)
            
            # Forecast the mortality rates
            mxt_hat = np.exp(self.params['ax'].reshape(-1,1) + 
                             np.einsum('i,j->ij', self.params['bx'], kt_forecast))
            
            # Filter the forecasted rates to only include the prediction horizon
            mxt_hat = mxt_hat[:, :]
            
            # If the true value is given            
            if(mxt_true is not None and mxt_true.shape[1] == horizon):
                residuals = mxt_true - mxt_hat
            else:
                residuals = np.array(0)

            return [kt_forecast, np.arange(self.idx_end+1, self.idx_end+horizon+1), mxt_hat, residuals]
            
    def residuals(self):
        """
        Calculate and return the residuals (raw errors) of the training data
        Returns:
            A vector of residuals calculated as (real - model) for the fitted data
        """
        # If the model has not been trained
        if(self.params == None):
            return None
        
        # reconstruction of parameters to mortality rates
        # outer product between bx and kt, then add ax to all ages (rows) for each year t (column)
        mxt_hat = np.exp(self.params['ax'].reshape(-1,1) + 
                            np.einsum('i,j->ij', self.params['bx'], self.params['kt']))
        return self.mxt - mxt_hat
    
class Rh():    
    def __init__(self, cohort_exclude_start, cohort_exclude_end):
        self.cohort_exclude_start = cohort_exclude_start
        self.cohort_exclude_end = cohort_exclude_end
        
    def fit(self, mxt, ext, year_start=0, year_end=0):
        """
        Fit the model with the given data
        Args:  
            mxt: a matrix of mortality rates with ages for rows and years for columns
            year_start: (optional) the first year of the fitted data, used to construct index for plotting (if given)
            year_end: (optional) the last year of the fitted data, used to construct index for plotting (if given)
        Returns:
        """
        # Store the data to calculate residuals
        self.mxt = mxt
        self.ext = ext

        # Determine the index (year) of the data (if givem)
        self.idx_start = year_start
        self.idx_end = year_end
        if(self.idx_end == 0):
            self.idx_end = mxt.shape[1]
        
        # Estimations of Lee-Carter Parameters
        [ax_hat, bx_1_hat, kt_hat, bx_2_hat, gz_hat, _] = rh_poisson_est(mxt=self.mxt, Ext=self.ext, 
                                                                         Ns1=self.cohort_exclude_start, 
                                                                         Ns2=self.cohort_exclude_end)
        self.params = {'ax': ax_hat,
                       'bx_1': bx_1_hat, 
                       'kt': kt_hat,
                       'bx_2': bx_2_hat,
                       'gz': gz_hat}

    def predict(self, horizon=0, order_kt=(0, 1, 0), order_gz=(1, 1, 0), mxt_true=None, ext_true=None):
        """
        Predict mortality rates a few years (horizon) ahead of the fitted data. 
        The default horizon is 0, which will predict mortality rates in the same years as the fitted data.
        Args:  
            horizon: the number of future years predicted
            order_kt: (default: 0, 1, 0) the ARIMA (p, d, q) parameters to forecast the time parameter kappa_t 
            order_gz: (default: 1, 1, 0) the ARIMA (p, d, q) parameters to forecast the cohort parameter gamma_z
            mxt_true: the true mortality rates for future years
            ext_true: the exposures for future years

        Returns:
            A list of numpy array containing the following information (in order):
                - The forecasted kappa_t
                - The forecasted gamma_z
                - The index (or year) of the forecasted parameters
                - The predicted mortality rates for all considered ages as rows and future years as columns
                * The lower bound of the predicted mortality rates
                * The upper bound of the predicted mortality rates
                - The error (raw difference) of the predicted mortality rates (if y_true is given or horizon=0)
        """
        # If horizon = 0, predict the mortality rates of the fitted data
        if(horizon == 0):
            # reconstruction of parameters to mortality rates
            mxt_hat, _, _ = get_mxt_dxt_diff(self.mxt*self.ext, self.ext, self.params['ax'], self.params['bx_1'], self.params['kt'], 
                                             self.params['bx_2'], self.params['gz'], 
                                             self.cohort_exclude_start, self.cohort_exclude_end)    
            return [self.params['kt'], self.params['gz'], np.arange(self.idx_start, self.idx_end+1), mxt_hat, self.mxt - mxt_hat]
        
        # If horizon > 0, predict the mortality rates of future years ahead of the fitted data
        else:
            # First make an ARIMA model
            if(order_kt is None):
                model_kt = pm.auto_arima(self.params['kt'],                       
                                         seasonal=False,  # TRUE if seasonal series
                                         d=None,             # let model determine 'd'
                                         test='adf',         # use adftest to find optimal 'd'
                                         start_p=0, start_q=0, # minimum p and q
                                         max_p=7, max_q=7, # maximum p and q
                                         D=None,             # let model determine 'D'
                                         trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)             
                self.order_kt = model_kt.order 
                model_gz = pm.auto_arima(self.params['gz'],
                                         seasonal=False,  # TRUE if seasonal series
                                         d=None,             # let model determine 'd'
                                         test='adf',         # use adftest to find optimal 'd'
                                         start_p=0, start_q=0, # minimum p and q
                                         max_p=7, max_q=7, # maximum p and q
                                         D=None,             # let model determine 'D'
                                         trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)             
                self.order_gz = model_gz.order
                
                # Forecast the time parameter kt according to ARIMA(order)            
                kt_forecast = model_kt.predict(n_periods=horizon)
                gz_forecast = model_gz.predict(n_periods=horizon)
            else:
                model_kt = ARIMA(self.params['kt'], order=order_kt, trend='t')
                self.order_kt = order_kt
                model_kt = model_kt.fit()

                model_gz = ARIMA(self.params['gz'], order=order_gz, trend='t')
                self.order_gz = order_gz
                model_gz = model_gz.fit()
                
                # Forecast the time parameter kt according to ARIMA(order)            
                kt_forecast = model_kt.forecast(steps=horizon)
                gz_forecast = model_gz.forecast(steps=horizon)
                        
            kt_all = np.append(self.params['kt'], kt_forecast)
            gz_all = np.append(self.params['gz'], gz_forecast)
            
            # Forecast the mortality rates
            mxt_hat = get_mxt_cohort(self.params['ax'], self.params['bx_1'], kt_forecast, 
                                             self.params['bx_2'], gz_forecast, 
                                             self.cohort_exclude_start, self.cohort_exclude_end)                            
            
            # If the true value is given, calculate residuals      
            if(mxt_true is not None and mxt_true.shape[1] == horizon):
                residuals = mxt_true - mxt_hat
            else:
                residuals = np.array(0)

            return [kt_forecast, gz_forecast, np.arange(self.idx_end+1, self.idx_end+horizon+1), mxt_hat, residuals]
            
    def residuals(self):
        """
        Calculate and return the residuals (raw errors) of the training data
        Returns:
            A vector of residuals calculated as (real - model) for the fitted data
        """
        # If the model has not been trained
        if(self.params == None):
            return None
        
        # reconstruction of parameters to mortality rates
        # outer product between bx and kt, then add ax to all ages (rows) for each year t (column)
        mxt_hat = np.exp(self.params['ax'].reshape(-1,1) + 
                            np.einsum('i,j->ij', self.params['bx'], self.params['kt']))
        return self.mxt - mxt_hat