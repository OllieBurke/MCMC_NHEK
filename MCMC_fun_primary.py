#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:32:28 2019

@author: Ollie
"""
"""
Created on Tue Oct 29 15:17:56 2019

@author: Ollie
"""
"""
The MCMC appears to be working but appears to yield slightly biased results. 
I don't understand why this is happening and don't have a huge intiuition as 
to how to fix this, if it is even a problem. 
"""
import numpy as np
from scipy import integrate

def InnerProduct(sig1,sig2,freq_bin,delta_t,PSD):
    """
    This function is used to compute the inner product between two signals
    sig1 and sig2. 
    """
    n_f = len(freq_bin) # Compute length of positive frequency components
    N = len(sig1)       # Compute length of time series
    fft_1 = np.fft.fft(sig1)  # Compute dft of sig1
    fft_2 = np.fft.fft(sig2)  # Compute dft of sig2
    # Below we return in the inner product of two signals sig1 and sig2.
    return np.real(sum((fft_1)[0:n_f] * np.conj(fft_2)[0:n_f])/(PSD * N/(4*delta_t)))

def Normalise(SNR,sig1,freq_bin,delta_t,PSD):
    """
    This function is used to normalise the amplitude of the signal so that 
    we get SNR = 1.
    """
    Normalise_Factor = InnerProduct(sig1,sig1,freq_bin,delta_t,PSD)
    return (SNR / np.sqrt(Normalise_Factor)) * sig1

a = 0.998
mu = 10
M_exact = 10**6
D = 1
phi = 3
SNR = 2000
rinit = 1.32

signal,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a,mu,M_exact,phi,D,rinit)

sigma = np.sqrt((PSD/(2*delta_t)))

noise = np.random.normal(0,sigma,len(signal))
noise = 0

data = signal + noise

plt.plot(new_t, data, color='k')
plt.plot(new_t, signal, color='g')
plt.show()


printerval = 50
target_accept = 0.44
adapt_batch = 20
M_var_prop = 10**12
Ntotal = 100000
burnin = 50000  # These are the same parameters you had before Matt
#
mcmc = MCMC_EMRI(data, SNR, delta_t, Ntotal, burnin, printerval,
                 M_var_prop, adapt_batch, target_accept, sigma)  # Calculate chain
##






