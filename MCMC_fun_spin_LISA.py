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

def llike(pdgrm, freq, n_t, delta_t, PSD):
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

    variances = (n_t / (4. * delta_t)) * PSD
 
    return(-0.5 * sum(pdgrm / variances))
    
    
 
   
def lprior(a):
    '''
    Compute log prior
    NOTE: Only spin included so far
    Inputs:
    spin: spin parameter
    spin_a, spin_b: alpha and beta prior parameters for beta density
    
    We note here that this function is actually unnecessary in the code
    since we are using a uniform prior.
    '''
    spin_a = 1000
    spin_b = 10
    
    return (1-spin_a)*np.log(a) + (1 - spin_b)*np.log(1-a)

    
def lpost(pdgrm, freq, n, delta_t, a_prop,PSD):
    '''
    Compute log posterior
    '''
    return(lprior(a_prop) + llike(pdgrm, freq, n, delta_t, PSD))


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
   elif (n - m) > 0:
       signal_prop = np.pad(signal_prop, (0, n - m), 'constant')# Otherwise pad the signal
   else:
       signal_prop = signal_prop
                                                                 
   return signal_prop
   
   
    


#####
#####

    
def MCMC_EMRI(signal_true, data_freq, largest_length, new_t,freq_bin, SNR, delta_t, Ntotal, burnin, printerval,
              a_var_prop, adapt_batch, target_accept, PSD):
    '''
    MCMC
    '''
    n_t = len(signal_true)  # Sample size

    
    Nadaptive = burnin
    a_log_sd_prop = np.log(a_var_prop) / 2 
    
    
    
#    freq_bin = np.fft.rfftfreq(n_t,delta_t)
#    n_f = len(freq_bin)
#    
    # Initial value for spin
    
    a = []
    a.append(0.999999)  # CAUTION: INITIAL VALUE
    
    mu = 10  # Same parameter values as before...
    M = 1e7
    D = 1
    phi = 0
    SNR = 20
    rinit = 2

    
    Normalise,GW_pad,_,_,r,new_t,Interpolating_Eps_Inf_Functions,delta_t,last_index = Un_Normed(a[0],mu,M,phi,D,rinit)
    # Exctract a new signal
    signal_init = (SNR/np.sqrt(Normalise))*GW_pad  # Normalise the signal for SNR = 20
    signal_init = Pad_Largest_Length(largest_length,signal_init)  # Pad to the longest length a = 1-10^-8

    
    signal_init_freq = FFT(signal_init)  # Calculate the fast fourier trnsform (centred)

    pdgrm = abs(data_freq - signal_init_freq)**2      # Compute periodogram of noise 
                                                      #(data - signal) in frequency doamin     
        
    # Initial value for log posterior
    lp = []
    lp.append(lpost(pdgrm, freq_bin, n_t, delta_t, a[0],PSD)) # append first value
                                                              # to log posterior.
    
    # Run MCMC
    for i in range(1, Ntotal):

        if i % printerval == 0:
            print("Iteration", i, "Proposal Variance", a_var_prop, 
                  "Spin is ", a[i - 1])
            
        #####
        # Adaptation
        #####
        if ((i < Nadaptive) and (i > 0) and (i % adapt_batch == 0)):
            a_log_sd_prop,a_var_prop = adapt_MH_proposal(i, a, a_log_sd_prop, adapt_batch, target_accept)

        #####
        
        # Previous values
        
        a_prev = a[i - 1]
        lp_prev = lp[i - 1]
        
        # Propose spin and do accept/reject step        
        a_prop = a_prev + np.random.normal(0, np.sqrt(a_var_prop), 1)[0]
        while a_prop < 0.999 or a_prop > 1-10**-8:
            """
            Here we stop the propsed values of spin going before and after
            a certain value. We don't want a spin a < 0.999 because then the
            interpolant won't give good results. We don't want a spin higher than
            a = 1-10^{-8} because this spin is stupid.
            """
            a_prop = a_prev + np.random.normal(0, np.sqrt(a_var_prop), 1)[0]

        # Proppose a new signal which has been normalised with SNR specified.
        signal_prop,new_t,delta_t,freq_bin,_,Normalise,last_index,max_index = GW_Normed(SNR,a_prop,mu,M,phi,D,rinit)
        signal_prop = Pad_Largest_Length(largest_length,signal_prop)   # pad to the longest length

        signal_prop_freq = FFT(signal_prop) # Compute signal_prop in frequency domain
        # Compute periodigram
        pdgrm_prop = abs(data_freq - signal_prop_freq)**2 
        # Compute log posterior
        lp_prop = lpost(pdgrm_prop, freq_bin, n_t, delta_t, a_prop, PSD)
        
        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
            a.append(a_prop)  # Accept and mu_prop is sampled
            lp.append(lp_prop)  # Accept new log posterior value
        else:  # Reject
            a.append(a_prev)  # reject mu_prop and use old value for next iteration
            lp.append(lp_prev)  # similar but with the log posterior
        # Append new value to the end of list
            
    return(a)
    


