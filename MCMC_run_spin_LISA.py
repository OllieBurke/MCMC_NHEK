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



# =============================================================================
# Parameters
# =============================================================================
Distance_sec,Msun_sec = units() # Extract "units". Don't worry about this.
EpsFun = Extrapolate(1-10**-9)  # Extract the extrapolating function for the relativistic
Interpolating_Eps_Inf_Functions = ExtrapolateInf_All(1-10**-9)

SNR = 20
a_exact = 0.9999  # This is the spin we want to find
mu = 10  # This is the secondary mass  
M = 1e7  # This is the primary mass
phi = 0  # This is the initial phase.
D = 0  # D here measures deviation from distance such that SNR = 20. This is important.

rinit = risco(0.999,1,1)[0]

a_max = 1-10**-8  # Maximum spin that we will consider in our MCMC
r,t,delta_t = Radial_Trajectory(a_max,mu,M,rinit,EpsFun)  # We compute the radial trajectory for the 
                                   # largest spin so that we pad all of our 
                                   # signals to this length

n_t = len(zero_pad(r))  # Calculate the largest length of the signal in time domain


r,t,delta_t = Radial_Trajectory(a_exact,mu,M,rinit,EpsFun)

fs = 1 / delta_t  # Sampling rate (measured in 1/seconds)
nyquist = fs / 2  # Nyquist frequency 
    
if n_t % 2:  # Odd
    n_f = (n_t - 1) // 2  # Note // rather than / for integer division!
else:  # Even
    n_f = n_t // 2 + 1        
freq_bin = np.linspace(0, np.pi, n_f) * nyquist / np.pi # In units of Hz. 
freq_bin = np.delete(freq_bin,0)        # Remove the zeroth frequency bin
n_f -= 1    # We remove the zeroth frequency bin so take 1 away from length 
            #in frequency domain

PSD = PowerSpectralDensity(freq_bin)  # Compute PSD


signal_true = signal(SNR,a_exact,mu,M,phi,D,rinit,Interpolating_Eps_Inf_Functions,r,
                     t,delta_t,freq_bin,PSD,n_f,n_t,Distance_sec)

variance_freq = len(signal_true)*PSD/(4*delta_t) # we have removed the zeroth frequency bin

noise_freq = np.array([np.random.normal(0,np.sqrt(variance_freq[i]),1)[0] for i in range(0,n_f)])
# Compute noise of signal in the frequency domain
signal_freq = FFT(signal_true)  # Compute the (centred) fast fourier transform
data_freq = signal_freq + noise_freq  # Compute data in the frequency domain





printerval = 50  # every 50 iterations we print what the proposal variance is 
                 # and our last sampled value.
target_accept = 0.44  # Single parameter so want to accept 44% of the time.
adapt_batch = 20  # We adapt the proposal variance every 20 iterations
a_var_prop = 1e-8   # initial proposal variance for mu
Ntotal = 800000  # Perform 80,000 iterations
burnin = 30000  # first 30000 iterations are for burnin

chain = MCMC_EMRI(n_t, data_freq, freq_bin, SNR, 
                  delta_t, Ntotal, burnin, printerval,
                  a_var_prop, adapt_batch, target_accept, PSD,Distance_sec)  # Calculate chain

sampled_a = chain[burnin:]  # Remove burnin values. Sampled from posterior.






