#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains the functions used to sample the secondary mass
from a posterior formed using the whittle likelihood. It appears to work...
"""


import numpy as np
import scipy as sp
import random as rd
import matplotlib.pyplot as plt

def llike(pdgrm, freq, n_t, delta_t, sigma):
    """
    Computes log (Whittle) likelihood 
    Assumption: Known PSD otherwise need additional term
    Requires: Function called lisa_psd()
    Inputs:
    pdgrm: periodogram 
    freq: frequencies
    n_t: length of time series
    deltat: sampling interval
    sigma: standard deviation of noise in time domain
    """
    psd = lisa_psd(freq,delta_t,sigma)
    variances = (n_t / (4. * delta_t)) * psd
 
    return(-0.5 * sum(pdgrm / variances))
    
    
def lisa_psd(freq, delta_t, sigma):
    '''
    Test PSD
    '''
    return (20/3)*1e-40
 
   
def lprior(mu):
    '''
    Compute log prior
    NOTE: Only spin included so far
    Inputs:
    spin: spin parameter
    spin_a, spin_b: alpha and beta prior parameters for beta density
    
    We note here that this function is actually unnecessary in the code
    since we are using a uniform prior.
    '''
    mu_a = 5
    mu_b = 20
    
    if mu < mu_a or mu > mu_b:
        return -1e100  # - infinity...
    else:
        return np.log((mu_b - mu_a)**-1)

    
def lpost(pdgrm, freq, n, delta_t, mu_prop,sigma):
    '''
    Compute log posterior
    '''
    return(lprior(mu_prop) + llike(pdgrm, freq, n, delta_t, sigma))


def accept_reject(lp_prop, lp_prev):
    '''
    Compute log acceptance probability (minimum of 0 and log acceptance rate)
    Decide whether to accept (1) or reject (0)
    '''
    u = np.random.uniform(size = 1)  # U[0, 1]
    r = np.minimum(0, lp_prop - lp_prev)  # log acceptance probability
    if np.log(u) < r:
        return(1)  # Accept
    else:
        return(0)  # Reject
    
 
def accept_rate(parameter):
    '''
    Compute acceptance rate for a specific parameter
    Used to adapt the proposal variance in a MH sampler
    Input: parameter (sequence of samples of a parameter)
    '''
    rejections = 0
    for i in range(len(parameter) - 1):  # Count rejections
        rejections = rejections + (parameter[i + 1] == parameter[i])
    reject_rate = rejections / (len(parameter) - 1)  # Rejection rate
    return(1 - reject_rate)  # Return acceptance rate


def adapt_MH_proposal(i, parameter, log_sd_prop, 
                      adapt_batch, target_accept):
    '''
    See Roberts + Rosenthal (2008) - Only works for Normal proposals
    Adapt the proposal variance of a parameter in MH sampler
    Uses accept_rate function
    Inputs:
    i: Iteration number
    parameter: Parameter sequence
    log_sd_prop: Current proposal log SD to be adapted
    adapt_batch: Batch size to adapt
    target_accept: Target acceptance rate (0.44 for 1 parameter)
    '''
    parameter = np.asarray(parameter)  # Convert list to array
    batch = np.arange(i - adapt_batch, i)  # Index used to adapt proposal
    adapt_delta = np.minimum(0.1, 1 / (i ** (1 / 3)))  # Scale adaptation
    batch_parameter = parameter[batch]  # Subset of parameter sequence 
    batch_accept = accept_rate(batch_parameter)  # Acceptance rate
    log_sd_prop = log_sd_prop + ((batch_accept > target_accept) * 2 - 1) * adapt_delta # New proposal log SD
    var_prop = np.exp(2 * log_sd_prop)  # Convert to proposal variance
    return([log_sd_prop, var_prop])  # Return new proposal log SD and variance
    



def Same_Length_Arrays(n, signal_prop):
   '''
   This function always either truncated the template or zero pads the template
   to make sure that it is of the same length as the data. I do need to be
   careful about zero padding the signal though... working in the frequency domain
   this could be problematic
   '''
   m = len(signal_prop)
   if (n - m) < 0:  # If length of data is smaller than length of signal
       index_del = list(range(n, m))  # Calculate indices which we will delete from the signal
       signal_prop = np.delete(signal_prop,index_del)  # Delete parts of the propposed signal.
   else:
       signal_prop = np.pad(signal_prop, (0, n - m), 'constant')  # Otherwise pad the signal
                                                                  # with zeros.
   return signal_prop
   
   
    


#####
#####

    
def MCMC_EMRI(data, SNR, delta_t, Ntotal, burnin, printerval,
              mu_var_prop, adapt_batch, target_accept, sigma):
    '''
    MCMC
    '''
    n_t = len(data)  # Sample size
    fs = 1/delta_t
    nyquist = fs/2

    
    Nadaptive = burnin
    mu_log_sd_prop = np.log(mu_var_prop) / 2 
    
    # Frequencies in Hz
    if n_t % 2:  # Odd
        n_f = (n_t - 1) // 2  # Note // rather than / for integer division!
    else:  # Even
        n_f = n_t // 2 + 1        
    freq_bin= np.linspace(0, np.pi, n_f) * nyquist / np.pi
    
    
#    freq_bin = np.fft.rfftfreq(n_t,delta_t)
#    n_f = len(freq_bin)
#    
    # Initial value for spin
    
    mu = []
    mu.append(5)  # CAUTION: INITIAL VALU
    
    a = 0.998
    M = 1e6
    D = 1
    rinit = 1.34
    phi = 3

    # Find signal given initial parameters
    #signal_init = GW_Full(spin[0], 0.0001) 
    
    Normalise,GW_pad,_,PSD,r,new_t,Interpolating_Eps_Inf_Functions,delta_t,last_index = Un_Normed(a,mu[0],M,phi,D,rinit)
    
    signal_init = (SNR/np.sqrt(Normalise))*GW_pad
    signal_init = Same_Length_Arrays(n_t, signal_init)
       
    # Compute periodogram of noise (data - signal)
    pdgrm = abs(np.fft.fft(data - signal_init)[range(0, n_f)])**2  # Same normalisation as R    
        
    # Initial value for log posterior
    lp = []
    lp.append(lpost(pdgrm, freq_bin, n_t, delta_t, mu[0],sigma))
    
    # Run MCMC
    for i in range(1, Ntotal):
        
        if i % printerval == 0:
            print("Iteration", i, "Proposal Variance", mu_var_prop, 
                  "Secondary mass", mu[i - 1])
            
        #####
        # Adaptation
        #####
        if ((i < Nadaptive) and (i > 0) and (i % adapt_batch == 0)):
            mu_log_sd_prop,mu_var_prop = adapt_MH_proposal(i, mu, mu_log_sd_prop, adapt_batch, target_accept)

            
        #####
        
        # Previous values
        
        mu_prev = mu[i - 1]
        lp_prev = lp[i - 1]
        
        # Propose spin and do accept/reject step        
        mu_prop = mu_prev + np.random.normal(0, np.sqrt(mu_var_prop), 1)[0]
        if mu_prop < 0:
            mu_prop = mu_prev + np.random.normal(0, np.sqrt(mu_var_prop), 1)[0]
        # Proppose a new signal which has been normalised with SNR specified.
        signal_prop,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a,mu_prop,M,phi,D,rinit)
        
        signal_prop = Same_Length_Arrays(n_t, signal_prop)  # Same length

        # Compute periodigram
        pdgrm_prop = abs(np.fft.fft(data - signal_prop)[range(0, n_f)])**2 
        # Compute log posterior
        lp_prop = lpost(pdgrm_prop, freq_bin, n_t, delta_t, mu_prop, sigma)
        
        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
            mu.append(mu_prop)  # Accept and mu_prop is sampled
            lp.append(lp_prop)  # Accept new log posterior value
        else:  # Reject
            mu.append(mu_prev)  # reject mu_prop and use old value for next iteration
            lp.append(lp_prev)  # similar but with the log posterior
        # Append new value to the end of list
            
    return(mu)
    


