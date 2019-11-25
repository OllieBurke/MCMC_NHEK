#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:06:18 2019

@author: mattedwards
"""

def rnoise(N, deltat, psd):
    '''
    Function to generate noise in time domain given a PSD
    Assumes one sided PSD
    '''   
    
    deltaf = 1 / (N * deltat)  # Distance between fourier frequencies
    
    if N % 2:  # Odd
      halfN = (N - 1) // 2
      kappa = [0]
      kappa += [1] * halfN
      lamb = [1]
      lamb += [2] * (halfN - 1)
    else:  # Even
      halfN = N // 2 + 1
      kappa = [0]
      kappa += [1] * (halfN - 1)
      kappa += [0]
      lamb = [1]
      lamb += [2] * (halfN - 1)
      lamb += [1]
      
    kappa = np.asarray(kappa)
    lamb = np.asarray(lamb)
    
    freq = np.arange(0., halfN + 1) * deltaf  # Fourier frequencies
    
    #####
    # Generate random noise in Fourier domain
    #####
    
    # SD of Fourier frequencies
    sigmaf = np.sqrt(np.asarray(psd(freq)) / ((1 + kappa) * lamb)) 

    # Remove infinite and negative SDs and set to zero - omit
    if any(sigmaf == inf) or any(sigmaf < 0):
        sigmaf[~np.isfinite(sigmaf)] = 0
        sigmaf[sigmaf < 0] = 0    
    
    # Sample Normal random variables
    a = np.random.normal(0, sigmaf, halfN+1)
    b = np.random.normal(0, sigmaf, halfN+1) * kappa
    
    # Find real and imaginary components
    real = np.sqrt(N / deltat) * a
    imag = -np.sqrt(N / deltat) * b
    
    # Create two-sided vector
    real_rev = real[kappa > 0]
    real_rev = real_rev[::-1]
    imag_rev = imag[kappa > 0]
    imag_rev = -imag_rev[::-1]
    real = np.concatenate([real, real_rev])
    imag = np.concatenate([imag, imag_rev])
    
    # Complex noise vector in frequency domain
    noiseFT = real + complex(0, 1) * imag
    
    # Inverse FT to time domain
    noiseTS = np.fft.ifft(noiseFT) 
    noiseTS = noiseTS.real  # Only take real parts
 
    return(noiseTS)
    

#x = rnoise(10000000, 1, lisa_psd)
#plt.plot(x)
