#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:53:26 2019

@author: mattedwards
"""

import numpy as np
import scipy as sp
import random as rd
import matplotlib.pyplot as plt

def llike(pdgrm, n_t, delta_t, PSD):
    """
    Computes log (Whittle) likelihood 
    Assumption: Known PSD otherwise need additional term
    Inputs:
    pdgrm: periodogram 
    n_t: length of time series
    delta_t: sampling interval
    PSD: LISA PSD (precomputed vector from PowerSpectralDensity)
    """
    
    variances = (n_t / (4. * delta_t)) * PSD 
    
    return(-0.5 * sum(pdgrm / variances))
    
    
def lprior(a, M, mu, phi, D,
           a_alpha, a_beta):
    '''
    Compute log prior
    Inputs:
    a: spin parameter, [0, 1] support
    M: primary mass, (0, Inf) support
    mu: secondary mass, (0, Inf) support
    phi: initial phase, [0, 2*pi] support
    a_alpha, a_beta: alpha and beta parameters for beta prior
    
    TO DO: Add prior parameters. Currently Uniform.
    '''
    
    # log prior for spin
    lp_a =  (1 - a_alpha) * np.log(a) + (1 - a_beta) * np.log(1 - a)
    
    # log prior for primary mass
    lp_M = 0
    
    # log prior for secondary mass
    lp_mu = 0
    
    # log prior for initial phase
    lp_phi = 0
        
    # log prior for initial phase
    lp_D = 0
        
    # Compute log prior
    lp = lp_a + lp_M + lp_mu + lp_phi + lp_D
    
    return(lp)

    
def lpost(pdgrm, n_t, delta_t, PSD,
          a, M, mu, phi, D, a_alpha, a_beta):
    '''
    Compute log posterior
    '''
    return(lprior(a, M, mu, phi, D, a_alpha, a_beta) + 
           llike(pdgrm, n_t, delta_t, PSD))


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
       signal_prop = np.pad(signal_prop, (0, n - m), 'constant') # Otherwise pad the signal
   else:
       signal_prop = signal_prop
                                                                 
   return signal_prop
   

#####
#####

def MCMC_EMRI(data_f, 
              Ntotal, 
              burnin, 
              printerval = 10,
              adapt_batch = 50, 
              target_accept = 0.44, 
              a_alpha = 1,
              a_beta = 1,
              a_min = 1 - 1e-3, 
              a_max = 1 - 1e-8, 
              M_min = 1e7,
              M_max = 1e7,
              mu_min = 10,
              mu_max = 10,
              eta_min = 1e-7,
              eta_max = 1e-5,
              SNR = 20,
              a_var_prop = 1,
              M_var_prop = 1,
              mu_var_prop = 1,
              phi_var_prop = 1,
              D_var_prop = 1):
    '''
    Metropolis-within-Gibbs MCMC sampler
    Arguments:
        TO DO
        data_freq: data in frequency domain (complex-valued)
        freq: etc.
        
    '''
    
    
    
    #####
    # Ollie's Preamble
    # TO DO: Tidy up

    EpsFun = Extrapolate(1-10**-9)  # Extract the extrapolating function for the relativistic
    Interpolating_Eps_Inf_Functions = ExtrapolateInf_All(1-10**-9)
    Distance_sec = units()[0]
    MSun_sec = units()[1]
    
    rinit = 1.6
    

    
    r_isco_max = risco(a_max)     # Calculate smallest ISCO.
                                            # This is the lowest radii that the particle
                                            # could ever achieve. 
    f_max = (11 * np.sqrt(Omega(r_isco_max, a_max, 0)) / (2 * np.pi * M_min * Msun_sec))  
    fs = 2 * f_max     # Set the sample rate equal to twice the highest frequency.
    nyquist = f_max    # We can change if needed
    delta_t = 1 / fs   # Calculate our samping rate.
    
    # Find maximum time series length    

    r, t = Radial_Trajectory(a_max, mu_min, M_max, rinit, EpsFun, a_max, delta_t)
    n_t = len(zero_pad(t))

    if n_t % 2:  # Odd
        n_f = (n_t - 1) // 2  # Note // rather than / for integer division!
    else:  # Even
        n_f = n_t // 2 + 1        
    
    freq = np.linspace(0, np.pi, n_f) * nyquist / np.pi # In units of Hz. 
    freq = np.delete(freq,0)        # Remove the zeroth frequency bin
    n_f -= 1    # We remove the zeroth frequency bin so take 1 away from length 
                #in frequency domain
    
    PSD = PowerSpectralDensity(freq)  # Compute PSD


    
    #####
    
    # Make data_f compatable
    
    
    #####
    
    # Open parameter objects
    a = []
    M = []
    mu = []
    phi = []
    D = []
    
    # Append initial values to parameter objects
    # CAUTION
    a.append(1-10**-6)  
    M.append(1e7)
    mu.append(10)
    phi.append(0)
    D.append(0)
    
    # Adapt for burnin length
    Nadaptive = burnin
    
    # Convert proposal variances to log scale for adapation
    a_log_sd_prop = np.log(a_var_prop) / 2 
    M_log_sd_prop = np.log(M_var_prop) / 2
    mu_log_sd_prop = np.log(mu_var_prop) / 2
    phi_log_sd_prop = np.log(phi_var_prop) / 2
    D_log_sd_prop = np.log(D_var_prop) / 2

    
    # Generate initial signal
    # r: radial trajectory
    # t: associated time (in seconds)
    r,t = Radial_Trajectory(a[0], mu[0], M[0], 
                             rinit, EpsFun, a_max, delta_t)    
    signal_init = signal(SNR, 
                         a[0], mu[0], M[0], phi[0],
                         D[0], rinit, FluxInf,
                         r, t, delta_t, freq, PSD, 
                         n_f, n_t, Distance_sec)

    # Calculate the fast fourier transform (centred)
    signal_init_f = FFT(signal_init)  

    # Compute periodogram of noise 
    pdgrm = abs(data_f - signal_init_f)**2     
                                                      
    # Initial value for log posterior
    lp = []
    lp.append(lpost(pdgrm, n_t, delta_t, PSD,
                    a[0], M[0], mu[0], phi[0], D[0],
                    a_alpha, a_beta))  # Append first value
    
    lp_store = lp[0]  # Create log posterior storage to be overwritten
                 
    #####                                                  
    # Run MCMC
    #####
    for i in range(1, Ntotal):

        if i % printerval == 0:
#            print("i =", i,
#                  ", M =", M[i - 1], 
#                  ", mu =", mu[i - 1], 
#                  ", a = ", a[i - 1],
#                  ", phi = ", phi[i - 1],
#                  ", M_sd =", np.sqrt(M_var_prop),
#                  ", mu_sd =", np.sqrt(mu_var_prop),
#                  ", a_sd =", np.sqrt(a_var_prop),
#                  ", phi_sd =", np.sqrt(phi_var_prop))
#                    print("i =", i,
#                  ", D =", D[i - 1], 
#                  ", a_sd =", np.sqrt(D_var_prop))
            print("i = ", i, "a = ",a[i-1], "proposal sd is", np.sqrt(a_var_prop))
            
        #####
        # Adaptation
        #####
        if ((i < Nadaptive) and (i > 0) and (i % adapt_batch == 0)):
            a_log_sd_prop, a_var_prop = adapt_MH_proposal(i, 
                                                          a, 
                                                          a_log_sd_prop, 
                                                          adapt_batch, 
                                                          target_accept)
#            M_log_sd_prop, M_var_prop = adapt_MH_proposal(i, 
#                                                          M, 
#                                                          M_log_sd_prop, 
#                                                          adapt_batch, 
#                                                          target_accept)
#            mu_log_sd_prop, mu_var_prop = adapt_MH_proposal(i, 
#                                                            mu, 
#                                                            mu_log_sd_prop, 
#                                                            adapt_batch, 
#                                                            target_accept)
#            phi_log_sd_prop, phi_var_prop = adapt_MH_proposal(i, 
#                                                              phi, 
#                                                              phi_log_sd_prop, 
#                                                              adapt_batch, 
#                                                              target_accept)
#            D_log_sd_prop, D_var_prop = adapt_MH_proposal(i, 
#                                                          D, 
#                                                          D_log_sd_prop, 
#                                                          adapt_batch, 
#                                                          target_accept)
        ####

        #####
        # Step 1: Sample spin, a
        #####
        
        lp_prev = lp_store  # Call previous stored log posterior
        
        # Propose spin and do accept/reject step   
        # Note: a in [a_min = 1-1e-3, a_max = 1-1e-8]
        a_prop = a[i - 1] + np.random.normal(0, np.sqrt(a_var_prop), 1)[0]
        while a_prop < a_min or a_prop > a_max:
            a_prop = a[i - 1] + np.random.normal(0, np.sqrt(a_var_prop), 1)[0]

        # Propose a new signal which has been normalised with SNR specified.

        r, t = Radial_Trajectory(a_prop, mu[i - 1], M[i - 1], 
                                    rinit, EpsFun, a_max, delta_t)    
        signal_prop = signal(SNR, a_prop, mu[i - 1], M[i - 1], phi[i - 1],
                             D[i -1], rinit, FluxInf,
                             r, t, delta_t, freq, 
                             PSD, n_f, n_t, Distance_sec)
        signal_prop_f = FFT(signal_prop)             # Convert to freq domain
        pdgrm_prop = abs(data_f - signal_prop_f)**2  # Compute periodigram
        
        # Compute log posterior
        lp_prop = lpost(pdgrm_prop, n_t, delta_t, PSD,
                        a_prop, M[i - 1], mu[i - 1], phi[i - 1], D[i - 1],
                        a_alpha, a_beta)
                
        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
            a.append(a_prop)  
            lp_store = lp_prop  # Overwrite lp_store
        else:  # Reject
            a.append(a[i - 1])  
#        a.append(a[0])        

        
#        ####
#         Step 2: Sample primary mass, M
#        ####
#
#        lp_prev = lp_store  # Call previous stored log posterior
#        
#        # Propose primary mass and do accept/reject step   
#        # Note: M in [M_min = 1e5, M_max = 1e8]
#        # Also: eta = mu / M in [eta_min = 1e-8, eta_max = 1e-4]
#        M_prop = M[i - 1] + np.random.normal(0, np.sqrt(M_var_prop), 1)[0]
#        eta = mu[i - 1] / M_prop
#        while M_prop < M_min or M_prop > M_max:
#            M_prop = M[i - 1] + np.random.normal(0, np.sqrt(M_var_prop), 1)[0]
#            eta = mu[i - 1] / M_prop
#
#        # Propose a new signal which has been normalised with SNR specified.
#        r,t = Radial_Trajectory(a[i], mu[i - 1], M_prop, 
#                                 rinit, EpsFun, a_max, delta_t)    
#        signal_prop = signal(SNR, a[i], mu[i - 1], M_prop, phi[i - 1],
#                             D, rinit, FluxInf,
#                             r, t, delta_t, freq, 
#                             PSD, n_f, n_t, Distance_sec)
#        signal_prop_f = FFT(signal_prop)             # Convert to freq domain
#        pdgrm_prop = abs(data_f - signal_prop_f)**2  # Compute periodigram
#        
#        # Compute log posterior
#        lp_prop = lpost(pdgrm_prop, n_t, delta_t, PSD,
#                        a[i], M_prop, mu[i - 1], phi[i - 1],
#                        a_alpha, a_beta)
#                
#        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
#            M.append(M_prop)  
#            lp_store = lp_prop  # Overwrite lp_store
#        else:  # Reject
#            M.append(M[i - 1])  
        M.append(M[0])
 
        ##
#         Step 3: Sample secondary mass, mu
        ##

#        lp_prev = lp_store  # Call previous stored log posterior
#        
#        # Propose secondary mass and do accept/reject step   
#        # Note: mu in [mu_min = ?, mu_max = ?]
#        # Also: eta = mu / M in [eta_min = 1e-8, eta_max = 1e-4]
#        mu_prop = mu[i - 1] + np.random.normal(0, np.sqrt(mu_var_prop), 1)[0]
##        eta = mu_prop / M[i]
##        while mu_prop < mu_min or mu_prop > mu_max or eta < eta_min or eta > eta_max:
#        while mu_prop < mu_min or mu_prop > mu_max:
#            mu_prop = mu[i - 1] + np.random.normal(0, np.sqrt(mu_var_prop), 1)[0]
##            eta = mu_prop / M[i]
#
#        # Propose a new signal which has been normalised with SNR specified.
#        r,t = Radial_Trajectory(a[i], mu_prop, M[i], 
#                                 rinit, EpsFun, a_max, delta_t)    
#        signal_prop = signal(SNR, a[i], mu_prop, M[i], phi[i - 1],
#                             D, rinit, FluxInf,
#                             r, t, delta_t, freq, 
#                             PSD, n_f, n_t, Distance_sec)
#        signal_prop_f = FFT(signal_prop)             # Convert to freq domain
#        pdgrm_prop = abs(data_f - signal_prop_f)**2  # Compute periodigram
#        
#        # Compute log posterior
#        lp_prop = lpost(pdgrm_prop, n_t, delta_t, PSD,
#                        a[i], M[i], mu_prop, phi[i - 1],
#                        a_alpha, a_beta)
#                
#        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
#            mu.append(mu_prop)  
#            lp_store = lp_prop  # Overwrite lp_store
#        else:  # Reject
#            mu.append(mu[i - 1])  
        mu.append(mu[0])

       ####
#         Step 4: Sample initial phase, phi
        ####

#        lp_prev = lp_store  # Call previous stored log posterior
#        
#        # Propose initial phase and do accept/reject step   
#        # Note: phi in [0, 2*pi]
#        phi_prop = phi[i - 1] + np.random.normal(0, np.sqrt(phi_var_prop), 1)[0]
#        while phi_prop < 0 or phi_prop > 2 * np.pi:
#            phi_prop = phi[i - 1] + np.random.normal(0, np.sqrt(phi_var_prop), 1)[0]
#
#        # Propose a new signal which has been normalised with SNR specified.
#        r, t = Radial_Trajectory(a[i], mu[i], M[i], 
#                                    rinit, EpsFun, a_max, delta_t)    
#        signal_prop = signal(SNR, a[i], mu[i], M[i], phi_prop,
#                             D, rinit, FluxInf,
#                             r, t, delta_t, freq, 
#                             PSD, n_f, n_t, Distance_sec)
#        signal_prop_f = FFT(signal_prop)             # Convert to freq domain
#        pdgrm_prop = abs(data_f - signal_prop_f)**2  # Compute periodigram
#        
#        # Compute log posterior
#        lp_prop = lpost(pdgrm_prop, n_t, delta_t, PSD,
#                        a[i], M[i], mu[i], phi_prop,
#                        a_alpha, a_beta)
#                
#        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
#            phi.append(phi_prop)  
#            lp_store = lp_prop  # Overwrite lp_store
#        else:  # Reject
#            phi.append(phi[i - 1]) 
        phi.append(phi[0])
#
#        # Propose D and do accept/reject step   
#
#        lp_prev = lp_store  # Call previous stored log posterior
#        
#        D_prop = D[i - 1] + np.random.normal(0, np.sqrt(D_var_prop), 1)[0]
##        while a_prop < a_min or a_prop > a_max:
##            a_prop = a[i - 1] + np.random.normal(0, np.sqrt(a_var_prop), 1)[0]
#
#        # Propose a new signal which has been normalised with SNR specified.
#
##        r, t = Radial_Trajectory(a[i - 1], mu[i - 1], M[i - 1], 
##                                    rinit, EpsFun, a_max, delta_t)    
#        signal_prop = signal(SNR, a[i - 1], mu[i - 1], M[i - 1], phi[i - 1],
#                             D_prop, rinit, FluxInf,
#                             r, t, delta_t, freq, 
#                             PSD, n_f, n_t, Distance_sec)
#        signal_prop_f = FFT(signal_prop)             # Convert to freq domain
#        pdgrm_prop = abs(data_f - signal_prop_f)**2  # Compute periodigram
#        
#        # Compute log posterior
#        lp_prop = lpost(pdgrm_prop, n_t, delta_t, PSD,
#                        a[i - 1], M[i - 1], mu[i - 1], phi[i - 1], D_prop,
#                        a_alpha, a_beta)
#                
#        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
#            D.append(D_prop)  
#            lp_store = lp_prop  # Overwrite lp_store
#        else:  # Reject
#            D.append(D[i - 1])  
        D.append(D[0])
 
        #####
        # Step 0: Compute log posterior 
        #####
        lp.append(lp_store)
        
    return a, M, mu, phi, lp,D
    


