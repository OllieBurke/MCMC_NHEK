#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:15:18 2019

@author: mattedwards
"""
"""
Good notes. Always make sure proposal variance is high enough so that the 
correct parameter can be found. 
"""
# Parameters to estimate
a_ex = 1-10**-6
M_ex = 1e7
mu_ex = 10
phi_ex = 0
D_ex = 0
# Other parameters
SNR = 20

# Ollie's stuff
Distance_sec,Msun_sec = units() # Extract "units". Don't worry about this.
EpsFun = Extrapolate(1-10**-9)  # Extract the extrapolating function for the relativistic
FluxInf = ExtrapolateInf_All(1-10**-9)

rinit = 1.6

a_max = 1-10**-8  # Maximum spin that we will consider in our MCMC
M_max = 1e7
M_min = 1e7
mu_min = 10
mu_max = 10

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

# Find radial trajectory of true signal
r,t = Radial_Trajectory(a_ex,mu_ex,M_ex,rinit,EpsFun,a_max, delta_t)

signal_true = signal(SNR,a_ex,mu_ex,M_ex,phi_ex,D_ex,rinit,FluxInf,r,
                     t,delta_t,freq,PSD,n_f,n_t,Distance_sec)

variance_freq = len(signal_true)*PSD/(4*delta_t) # we have removed the zeroth frequency bin

noise_freq = np.array([np.random.normal(0,np.sqrt(variance_freq[i]),1)[0] for i in range(0,n_f)])
# Compute noise of signal in the frequency domain
signal_freq = FFT(signal_true)  # Compute the (centred) fast fourier transform

data_f = signal_freq + noise_freq  # Compute data in the frequency domain



mcmc = MCMC_EMRI(data_f, 
              Ntotal = 50000, 
              burnin = 20000, 
              printerval = 50,
              adapt_batch = 50, 
              target_accept = 0.44, 
              a_var_prop = (1.34e-9)**2,
              M_var_prop = 1e13,
              mu_var_prop = 100,
              phi_var_prop = 1,
              D_var_prop = 0.1)


# Post-processing
print('starting value',rinit)
print('exact a is',a_ex)
#print('exact a is',M_ex)
#print('exact a is',mu_ex)


ap = mcmc[0]
ap = ap[burnin:Ntotal]
plt.plot(ap)
np.sqrt(np.var(ap))
np.quantile(ap, (0.025, 0.975))
plt.show()
plt.clf()

Mp = mcmc[1]
Mp = Mp[burnin:Ntotal]
plt.plot(Mp)
np.sqrt(np.var(Mp))
np.quantile(Mp, (0.025, 0.975))
plt.show()
plt.clf()
lp = mcmc[4]
lp = lp[burnin:Ntotal]
plt.plot(lp)
plt.show()
plt.clf()

sampled_M = mcmc[1][burnin:]
sampled_mu = mcmc[2][burnin:]





