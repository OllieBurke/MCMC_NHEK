#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:18:13 2019

@author: Ollie
"""

 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:47:52 2019

@author: Ollie
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
    pdgrm: periodogram (MAKE SURE CORRECT SCALE)
    freq: frequencies
    n: length of time series
    deltat: sampling interval
    sigma: standard deviation of noise in time domain
    """
    psd = (20/3)*1e-40 # The LISA PSD (function of frequency)
    variances = (n_t / (4. * delta_t)) * psd

    #return(-0.5 * sum(np.log(psd) + pdgrm / variances))    
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
    '''
    
    if mu < 5 or mu > 20:
        return -1e100
    else:
        return np.log(1/15)
    
def lprior(M):
    '''
    Compute log prior
    NOTE: Only spin included so far
    Inputs:
    spin: spin parameter
    spin_a, spin_b: alpha and beta prior parameters for beta density
    '''
    
    if M < 5*1e5 or M > 5*1e7:
        return -1e100
    else:
        return np.log((5*1e7 - 5*1e5)**-1)

    
def lpost(pdgrm, freq, n, delta_t, mu_prop,M_prop,sigma):
    '''
    Compute log posterior
    '''
#    return(lprior(M_prop) + lprior(mu_prop) + llike(pdgrm, freq, n, delta_t, sigma))
    return lprior(mu_prop) + lprior(M_prop) +  llike(pdgrm, freq, n, delta_t, sigma)


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
   if (n - m) < 0:
       index_del = list(range(n, m))
       signal_prop = np.delete(signal_prop,index_del)
   else:
       signal_prop = np.pad(signal_prop, (0, n - m), 'constant')
   return signal_prop
   
   
   
    


#####
#####

    
def MCMC_EMRI(data, SNR, delta_t, Ntotal, burnin, printerval,
              mu_var_prop, M_var_prop, adapt_batch, target_accept, sigma):
    '''
    MCMC
    '''
    n_t = len(data)  # Sample size
    fs = 1/delta_t
    nyquist = fs/2

    
    Nadaptive = burnin
    mu_log_sd_prop = np.log(mu_var_prop) / 2 
    M_log_sd_prop = np.log(M_var_prop)/2
    
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
    
    mu,M = [],[]
    
    mu.append(7)  # CAUTION: INITIAL VALU
    M.append(1e6)
    
    a = 0.998
    D = 1
    rinit = 1.34
    phi = 3

    # Find signal given initial parameters
    #signal_init = GW_Full(spin[0], 0.0001) 
    
    Normalise,GW_pad,_,PSD,r,new_t,Interpolating_Eps_Inf_Functions,delta_t,last_index = Un_Normed(a,mu[0],M[0],phi,D,rinit)
    
    signal_init = (SNR/np.sqrt(Normalise))*GW_pad
    signal_init = Same_Length_Arrays(n_t, signal_init)
       
    # Compute periodogram of noise (data - signal)
    pdgrm = abs(np.fft.fft(data - signal_init)[range(0, n_f)])**2  # Same normalisation as R    
        
    # Initial value for log posterior
    lp_mu,lp_M = [],[]
    lp_mu.append(lpost(pdgrm, freq_bin, n_t, delta_t, mu[0],M[0],sigma))
    lp_M.append(lp_mu[0])
    # Run MCMC
    for i in range(1, Ntotal):
        
        if i % printerval == 0:
            print("Iteration", i, "Mu_var_prop", mu_var_prop, 
                  "Secondary mass", mu[i - 1])
            print("M_var_prop",M_var_prop,"Primary mass", M[i-1])
            
        #####
        # Adaptation
        #####
        if ((i < Nadaptive) and (i > 0) and (i % adapt_batch == 0)):
            mu_log_sd_prop,mu_var_prop  = adapt_MH_proposal(i, mu, mu_log_sd_prop, adapt_batch, target_accept)
            M_log_sd_prop,M_var_prop = adapt_MH_proposal(i, M, M_log_sd_prop, adapt_batch, target_accept)
            
        #####
        
        # Previous values
        
        mu_prev = mu[i - 1]
        M_prev = M[i - 1]
        
        lp_mu_prev = lp_mu[i - 1]
        lp_M_prev = lp_M[i - 1]
        
        # Propose mu and do accept/reject step        
        mu_prop = mu_prev + np.random.normal(0, np.sqrt(mu_var_prop), 1)[0]
        
        signal_prop_mu,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a,mu_prop,M_prev,phi,D,rinit)       
        signal_prop_mu = Same_Length_Arrays(n_t, signal_prop_mu)
        pdgrm_prop_mu = abs(np.fft.fft(data - signal_prop_mu)[range(0, n_f)])**2 
        
        lp_mu_prop = lpost(pdgrm_prop_mu, freq_bin, n_t, delta_t, mu_prop,M_prev, sigma)
        
        if accept_reject(lp_mu_prop, lp_mu_prev) == 1:  # Accept
            mu.append(mu_prop)
            lp_mu.append(lp_mu_prop)
        else:  # Reject
            mu.append(mu_prev)
            lp_mu.append(lp_mu_prev)
        
        new_mu = mu[i] # next parameter
        M_prop = M_prev + np.random.normal(0,np.sqrt(M_var_prop),1)[0]
        
        signal_prop_M,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a,new_mu,M_prop,phi,D,rinit)       
        signal_prop_M = Same_Length_Arrays(n_t, signal_prop_M)
        pdgrm_prop_M = abs(np.fft.fft(data - signal_prop_M)[range(0, n_f)])**2 
        
        lp_M_prop = lpost(pdgrm_prop_M, freq_bin, n_t, delta_t, new_mu , M_prop, sigma)
        
        if accept_reject(lp_M_prop, lp_M_prev) == 1:  # Accept
            M.append(M_prop)
            lp_M.append(lp_M_prop)
        else:  # Reject
            M.append(M_prev)
            lp_M.append(lp_M_prev)

    full_chain_mu = mu
    chain_mu = full_chain_mu[burnin:]
    
    full_chain_M = M
    chain_M = full_chain_M[burnin:]
    
    return full_chain_mu,chain_mu,full_chain_M,chain_M
    


