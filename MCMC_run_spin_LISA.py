#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:40:57 2019

@author: Ollie
"""

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
Goooooood goood. Looks like it's starting to do something pretty good.

I need to make sure that I pad my data to the MAXIMUM length. Suppose I set
a criteria such that I only zero pad to a length of 1-10^{-9} length of signal.
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

def Pad_Largest_Length(largest_length,signal):
    return  np.pad(signal,(0,largest_length - len(signal)),'constant')
def Normalise(SNR,sig1,freq_bin,delta_t,PSD):
    """
    This function is used to normalise the amplitude of the signal so that 
    we get SNR = 1.
    """
    Normalise_Factor = InnerProduct(sig1,sig1,freq_bin,delta_t,PSD)
    return (SNR / np.sqrt(Normalise_Factor)) * sig1

# =============================================================================
# Parameters
# =============================================================================
a_exact = 0.9999  # Exact value of spin that we want to find
mu = 10  # Secondary mass
M = 1e7  # Primary mass
D = 1   # Arbitrary distance that we will tune so that we get SNR = 20
phi = 0  # Initial phase
SNR = 20 # SNR of signal
rinit = 2  # Initial raddi at which we start to integrate the solution
a_max = 1-10**-8  # Maximum spin that we will consider in our MCMC
r,t,delta_t = R(a_max,mu,M,rinit)  # We compute the radial trajectory for the 
                                   # largest spin so that we pad all of our 
                                   # signals to this length

largest_length = len(zero_pad(r))  # Calculate the largest length of the signal.

# =============================================================================
# Signal we wish to estimate and perform parameter estimation on.
# =============================================================================

signal,new_t,delta_t,_,_,Normalise,last_index,max_index = GW_Normed(SNR,a_exact,mu,M,phi,D,rinit)
# Extract the true signal.
n_t = len(signal)  # Length of signal in the time domain


signal = Pad_Largest_Length(largest_length,signal)  # Pad the signal to the largest length
freq_bin = np.delete(np.fft.rfftfreq(len(signal),delta_t),0) # Exctract fourier frequencies
PSD = PowerSpectralDensity(freq_bin)  # Calculate the PSD at the frequencies sampled above


variance_freq = n_t*PSD/(4*delta_t) # we have removed the zeroth frequency bin

noise_freq = np.array([np.random.normal(0,np.sqrt(variance_freq[i]),1)[0] for i in range(0,n_f)])
# Compute noise of signal in the frequency domain
signal_freq = FFT(signal)  # Compute the (centred) fast fourier transform
data_freq = signal_freq + noise_freq  # Compute data in the frequency domain





printerval = 50  # every 50 iterations we print what the proposal variance is 
                 # and our last sampled value.
target_accept = 0.44  # Single parameter so want to accept 44% of the time.
adapt_batch = 20  # We adapt the proposal variance every 20 iterations
a_var_prop = 1e-8   # initial proposal variance for mu
Ntotal = 20000  # Perform 30,000 iterations
burnin = 7500  # first 10,000 iterations are for burnin

chain = MCMC_EMRI(signal, data_freq, largest_length, new_t,freq_bin, SNR, delta_t, Ntotal, burnin, printerval,
              a_var_prop, adapt_batch, target_accept, PSD)  # Calculate chain

sampled_a = chain[burnin:]  # Remove burnin values. Sampled from posterior.






