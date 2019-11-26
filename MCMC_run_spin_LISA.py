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

#def InnerProduct(sig1,sig2,freq_bin,delta_t,PSD):
#    """
#    This function is used to compute the inner product between two signals
#    sig1 and sig2. 
#    """
#    n_f = len(freq_bin) # Compute length of positive frequency components
#    N = len(sig1)       # Compute length of time series
#    fft_1 = np.fft.fft(sig1)  # Compute dft of sig1
#    fft_2 = np.fft.fft(sig2)  # Compute dft of sig2
#    # Below we return in the inner product of two signals sig1 and sig2.
#    return np.real(sum((fft_1)[0:n_f] * np.conj(fft_2)[0:n_f])/(PSD * N/(4*delta_t)))

def Pad_Largest_Length(largest_length,signal):
    return  np.pad(signal,(0,largest_length - len(signal)),'constant')
#def Normalise(SNR,sig1,freq_bin,delta_t,PSD):
#    """
#    This function is used to normalise the amplitude of the signal so that 
#    we get SNR = 1.
#    """
#    Normalise_Factor = InnerProduct(sig1,sig1,freq_bin,delta_t,PSD)
#    return (SNR / np.sqrt(Normalise_Factor)) * sig1

def signal(SNR,a,mu,M,phi,D,rinit,Interpolating_Eps_Inf_Functions,r,t,delta_t,largest_length,freq_bin,PSD,n_f,n_t):
    """
    This function calculates the signal with the parameters mentioned above. Here 
    """
    un_normalised_signal = zero_pad(Waveform_All_Modes(r,t,a,mu,M,phi,D,Interpolating_Eps_Inf_Functions))
    GW_pad = Pad_Largest_Length(largest_length,un_normalised_signal)
    Normalise = Inner_Prod(GW_pad,GW_pad,delta_t,freq_bin,PSD,n_f,n_t) # Find normalising factor.
    return (SNR/np.sqrt(Normalise))*GW_pad

# =============================================================================
# Parameters
# =============================================================================
EpsFun = Extrapolate(1-10**-9)  # Extract the extrapolating function for the relativistic
Interpolating_Eps_Inf_Functions = ExtrapolateInf_All(1-10**-9)

SNR = 20
a_exact = 0.9999
mu = 10
M = 1e7
phi = 0
D = 1
rinit = 1.2

a_max = 1-10**-8  # Maximum spin that we will consider in our MCMC
r,t,delta_t = Radial_Trajectory(a_max,mu,M,rinit,EpsFun)  # We compute the radial trajectory for the 
                                   # largest spin so that we pad all of our 
                                   # signals to this length

largest_length = len(zero_pad(r))  # Calculate the largest length of the signal.
n_t = largest_length

r,t,delta_t = Radial_Trajectory(a_exact,mu,M,rinit,EpsFun)

fs = 1 / delta_t  # Sampling rate (measured in 1/seconds)
nyquist = fs / 2  # Nyquist frequency 
    
if largest_length % 2:  # Odd
    n_f = (largest_length - 1) // 2  # Note // rather than / for integer division!
else:  # Even
    n_f = largest_length // 2 + 1        
freq_bin = np.linspace(0, np.pi, n_f) * nyquist / np.pi # In units of Hz. 
freq_bin = np.delete(freq_bin,0)        # Remove the zeroth frequency bin
n_f -= 1    # We remove the zeroth frequency bin so take 1 away from length 
            #in frequency domain

PSD = PowerSpectralDensity(freq_bin)  # Compute PSD


signal_true = signal(SNR,a_exact,mu,M,phi,D,rinit,Interpolating_Eps_Inf_Functions,r,
                     t,delta_t,largest_length,freq_bin,PSD,n_f,n_t)

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
Ntotal = 20000  # Perform 30,000 iterations
burnin = 7500  # first 10,000 iterations are for burnin

chain = MCMC_EMRI(signal_true, data_freq, largest_length, freq_bin, SNR, 
                  delta_t, Ntotal, burnin, printerval,
                  a_var_prop, adapt_batch, target_accept, PSD)  # Calculate chain

sampled_a = chain[burnin:]  # Remove burnin values. Sampled from posterior.






