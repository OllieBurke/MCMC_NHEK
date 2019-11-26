#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:21:31 2019

@author: Ollie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:19:37 2019

@author: Ollie
"""
import scipy
from scipy import integrate
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import simps
from scipy.integrate import odeint
from math import *
from scipy import signal
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate
import os
import warnings

matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 

matplotlib.rcParams.update({'font.size': 14})
warnings.filterwarnings('ignore')  # Because of stiff differential equation.

def units():
    """
    Bloody units... throughout the entire code, we define everything in terms of
    seconds.
    """
    G = 6.6726e-11
    c = 299792458
    Msun = 1.9889e30
    Gpc = 1.02938e17
    Msun_sec = G*Msun/(c**3)  # All solar masses in seconds
    Distance_sec = (3.08567818585e25/c)  # Distances in seconds
    return Distance_sec,Msun_sec

def risco(a,m,M):
    '''
    This function computes the inner most stable circular orbit (ISCO)
    given the mass M of the black hole and its corresponding spin a.
    '''
    Z1 = 1. + ((1.-a**2)**(1/3))*((1.+a)**(1/3) + (1.-a)**(1/3))
    Z2 = (3.*a**2 + Z1**2)**(1/2)
    
    if a >= 0:       # Prograde (co-rotating) 
        r_isco = (3. + Z2 - ((3. - Z1)*(3.+Z1 + 2.*Z2))**(1/2))*M  # This evaluates the r_isco.
    else:            # Retrograde
        r_isco = (3. + Z2 + ((3. - Z1)*(3.+Z1 + 2.*Z2))**(1/2))*M
    return r_isco,a   # In later functions, r_isco, a

def diffeqn(t,r0,a,EpsFun,r_isco,eta):
    '''
    This is the differential equation dr0/dt = f(r0) that we wish to numerically integrate.
    '''
    if r0 == -inf or r0 == inf:

        return nan
    
    elif r0<r_isco:
        return nan
    else:       
        return -1*eta*EpsFun(r0)*(64/5)*((r0)**(3/2) + a)**(-10/3)*  ((1 - (3/r0) + 2*a*((1/r0)**(3/2)))**(3/2))/ ((1-(6/r0) + 8*a*(1/r0)**(3/2) - 3*(a/r0)**2)*((1/r0)**2))  

def Radial_Trajectory(a,mu,M,rinit,EpsFun):
    '''
    INPUTS: Spin a and initial radii to begin inspiral rinit
    
    OUTPUTS: inspiral radial trajectory r, Boyer Lindquist time t and sampling interval delta_t
    This function uses a standard python integrator to numerically integrate the differential equation
    above. It will return the values r against t alongside the interpolating/extrapolating 
    functions. Want to generalte inspiral from w0 = 6.714M since this will generate a year longh inspiral.    
    
    WARNING: outputs r_insp in units of M. t_insp in units of M/eta. MUST multiply by M * Msun_sec to find equivalent in seconds.
    '''


                             # correction values.

    r_isco = risco(a,1,1)[0] # Calculate the ISCO at a spin of a
    

    #rinit = 6.714          # Initial value 6.714M (boundary condition) (year long at a = 1-10**-9)
    
    eta = mu/M
    
    delta_t = 1 # Choose nice and small delta_t. This is in units of seconds. This is our sampling interval.

    t0 = 0 # initial time.

    r = ode(diffeqn, jac = None).set_integrator('lsoda', method='bdf')
    r.set_initial_value(rinit,t0).set_f_params(a,EpsFun,r_isco,eta)
#    one_day = 86400
    
    r_insp = []
    t_insp = []
    
    j = 0    
    while r.successful():
        # Integrate the ODE in using a while look. The second condition is a criteria to stop.
    #    print(r.t+dt, r.integrate(r.t+dt))< time step and solution r(t+dt).
        r_int = r.integrate(r.t+delta_t)[0]   # compute each integrated r value.

        if r_int<r_isco or isnan(r_int):  # Stop if we integrate past the ISCO
            break 
        t_insp.append(r.t+delta_t)  # Append timestep to a list.
        r_insp.append(r_int)

        if (r_insp[j] - r_insp[j-1]) > 0:
            del r_insp[-1]
            del t_insp[-1]
            break
        j+=1
#    print('Last radial coordinate is',r_insp[-1])
#    delta_t*=(1e6 * Msun_sec/eta)
#    t_insp *= (1e6 *Msun_sec/eta)  
    return np.array(r_insp),np.array(t_insp),delta_t

def PowerSpectralDensity(f):
    """
    Power Spectral Density for the LISA detector assuming it has been active for a year. 
    I found an analytic version in one of Niel Cornish's paper which he submitted to the arXiv in
    2018. I evaluate the PSD at the frequency bins found in the signal FFT.
    
    Am I definitely using the correct PSD? I am averaging over the sky so should I not be using
    a slightly different PSD? EDIT: Corrected, I think... but need to ask Jonathan
    where he got that (20/3) number (Dublin lecture notes). This is not obvious. 
    
    PSD obtained from: https://arxiv.org/pdf/1803.01944.pdf
    
    
    """
    sky_averaging_constant = (20/3) # Sky Averaged <--- I got this from Jonathan's notes but I need
                                    # to check where he got it...
    L = 2.5*10**9   # Length of LISA arm
    f0 = 19.09*10**-3    

    Poms = ((1.5*10**-11)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))  

    PSD = (sky_averaging_constant)* ((10/(3*L**2))*(Poms + (4*Pacc)/((2*np.pi*f))**4)*(1 + 0.6*(f/f0)**2) + Sc) # PSD


    return PSD
    

def double_fact(number):
    ''' 
    Quick function which computes double factorial.
    '''
    if number==0 or number==1:
        return 1
    else:
        return number*double_fact(number-2)
    
def Factor_E(m):
    '''
    This is the numerical factor which pre-multiplies the flux.
    '''
    # Constant factor in front of the flux for each harmonic
    return ((2*(m+1)*(m+2)*factorial(2*m+1)*m**(2*m+1))/((m-1)*((2**m)*factorial(m)*double_fact(2*m+1))**2))

def Omega(r,a,m):
    # Particles **dimensionless** angular momenta
    return ((r**(3/2) + a)**-1)**(2 + 2*m/3)
def Waveform(r,t,a,m,mu,M,phi,D,Interpolating_Eps_Inf_Functions):
    '''
    This code gives, analytically, the expression for a waveform at a particular mode.
    The waveform is the root mean square where we have averaged over the sky. As a result, 
    this is the (averaged) response that LISA would see. Or, that is my interpretation anyway.
    
    Waveform model found in: https://arxiv.org/pdf/gr-qc/0007074.pdf

    INPUTS: The inputs t_insp are measured in seconds. The orbital velocity is
    dimensionless. Note the sqrt on the Omega... this is due to th definition 
    above. Since t is in seconds, we di
    '''
    # Compute the waveform for each harmonic

    Distance_sec,Msun_sec  = units()    
    secondary = mu * Msun_sec
    primary = M * Msun_sec
    return (secondary/D) * ((2/(m+2))*np.sqrt(Factor_E(m+2)*Omega(r,a,m+2)* \
             Interpolating_Eps_Inf_Functions[m](r))/np.sqrt(Omega(r,a,0))) * np.sin((m+2)*(np.sqrt(Omega(r,a,0)))*t/primary + phi)

    
    
def Waveform_All_Modes(r,t,a,mu,M,phi,D,Interpolating_Eps_Inf_Functions):
    '''
    This function computes the (Un-normed) gravitational waveform 
    summing each of the individual harmonics (voices). It will build a waveform
    which is not normalised. Here we use 11 harmonics and add them.
    '''
    return Waveform(r,t,a,0,mu,M,phi,D,Interpolating_Eps_Inf_Functions) + \
        Waveform(r,t,a,1,mu,M,phi,D,Interpolating_Eps_Inf_Functions) + \
        Waveform(r,t,a,2,mu,M,phi,D,Interpolating_Eps_Inf_Functions) + \
        Waveform(r,t,a,3,mu,M,phi,D,Interpolating_Eps_Inf_Functions) + \
        Waveform(r,t,a,4,mu,M,phi,D,Interpolating_Eps_Inf_Functions) + \
        Waveform(r,t,a,5,mu,M,phi,D,Interpolating_Eps_Inf_Functions) + Waveform(r,t,a,6,mu,M,phi,D,Interpolating_Eps_Inf_Functions) + \
        Waveform(r,t,a,7,mu,M,phi,D,Interpolating_Eps_Inf_Functions) + \
        Waveform(r,t,a,8,mu,M,phi,D,Interpolating_Eps_Inf_Functions) + \
        Waveform(r,t,a,9,mu,M,phi,D,Interpolating_Eps_Inf_Functions) 
 
def zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data,(0,int((2**pow_2)-N)),'constant')

def FFT(signal):
    return np.delete(np.fft.rfftn(signal),0)

def Inner_Prod(signal1,signal2,delta_t,freq_bin,PSD,n_f,n_t):
    """ 
    Computes inner product of signal
    IMPORTANT: here we use the DISCRETE fourier transform rather than the 
    continuous fourier transform. 
    See 
    
    https://www.rdocumentation.org/packages/bspec/versions/1.5/topics/snr
    
    for the formula
    """
#    
#    n_f = len(freq_bin)   # length of signal in frequency doamin
#    n_t = len(signal1)    # length of signal in the time domain

    
    FFT_Sig1 = FFT(signal1)
    FFT_Sig2 = FFT(signal2)
    
    Inner_Product = 4*delta_t*sum(np.real((FFT_Sig1 * np.conjugate(FFT_Sig2))) / (n_t * PSD))
    
    return Inner_Product

def All_Modes_GW_Waveform(a,mu,M,phi,D,rinit):
    '''
    This function creates the un normed gravitational waveform. 
    INPUTS: a: spin
            mu: secondary mass
            M: primary mass
            phi: initial phase
            D: distance (set to be 1) and we will normalise the signal later
            rinit: Initial radius to begin inspiral
    '''
    
    Interpolating_Eps_Inf_Functions = ExtrapolateInf_All(a)
    # Extract interpolating functions
    Distance_sec,Msun_sec = units()  # Extract helpful units
    # Extract the two interpolating functions for the fluxes

    r,t,delta_t = R(a,mu,M,rinit) # Compute trajectories.

    GW = Waveform_All_Modes(r,t,a,mu,M,phi,D,Interpolating_Eps_Inf_Functions)
    last_index = len(GW)  # Find the length of the time series GW
    GW_pad = zero_pad(GW)  # Zero pad the data so it is of power of two length
    new_t = np.concatenate([t,[t[-1] + i*delta_t for i in range(1,(len(GW_pad) - len(GW) + 1))]])
    # Find new times for zero padded data    
#    delta_t*=(M * Msun_sec/eta)   # Find sampling interval in seconds. We do 
                                  # this so that we are not caught out when
                                  # we take fourier transforms etc.
    

    return GW_pad,r,new_t,Interpolating_Eps_Inf_Functions,delta_t,last_index




def Un_Normed(a,mu,M,phi,D,rinit):
    '''
    This function computes the normalising factor so that my signal has
    SNR equal to whatever I specify. It essentially calculates the distance
    so that my signal has the SNR of my choice..
    '''
    GW_pad,r,new_t,Interpolating_Eps_Inf_Functions,delta_t,last_index = All_Modes_GW_Waveform(a,mu,M,phi,D,rinit) # Extract GW
    
    n = len(GW_pad)   # Sample size in the time domain
    fs = 1 / delta_t  # Sampling rate (measured in 1/seconds)
    nyquist = fs / 2  # Nyquist frequency 
        
    if n % 2:  # Odd
        n_f = (n - 1) // 2  # Note // rather than / for integer division!
    else:  # Even
        n_f = n // 2 + 1        
    freq_bin = np.linspace(0, np.pi, n_f) * nyquist / np.pi # In units of Hz. 
    freq_bin = np.delete(freq_bin,0)        # Remove the zeroth frequency bin


    PSD = PowerSpectralDensity(freq_bin)  # Compute PSD
    
    Normalise = Inner_Prod(GW_pad,GW_pad,delta_t,freq_bin,PSD) # Find normalising factor.

    return Normalise,GW_pad,freq_bin,PSD,r,new_t,Interpolating_Eps_Inf_Functions,delta_t,last_index


def GW_Normed(SNR,a,mu,M,phi,D,rinit):
    '''
    Generates the normalised signal for SNR = whatever I want. 
    '''  
    SNR = 20
    a = 1-10**-9
    mu = 10
    M = 1e7
    phi = 0
    D = 1
    rinit = 2  # 1 year with eta = 10^{-6}, M = 1e7 <-- better for NHEK.

    Normalise,GW_pad,freq_bin,PSD,r,new_t,Interpolating_Eps_Inf_Functions,delta_t,last_index = Un_Normed(a,mu,M,phi,D,rinit)
    # Calculate the Normalising factor
    Distance_sec,Msun_sec = units()  # Extract helpful units
    # Read in distance in seconds.
    Distance = (((SNR/np.sqrt(Normalise))**-1)) / (Distance_sec)
    
    # Calculate the distance in Gpc
    
    #    GW_check = (1/(Distance*Distance_sec)) * GW_pad
    
    GW_pad *= (SNR / np.sqrt(Normalise))  # The signal has SNR for what I specified.
    max_index = np.argmax(GW_pad)
    eta = mu/M
    #    plt.plot(new_t[0:last_index]*eta,GW_pad[0:last_index],'darkviolet')
    #    plt.ylabel(r'$h$')
    #    plt.xlabel(r'$\tilde{t}\times\eta$')
    #    plt.title('Near-Extremal Waveform')
    
#    print('length of signal is',new_t[last_index]/(31536000))
    

    return GW_pad,new_t,delta_t,freq_bin,PSD,Normalise,last_index,max_index

def signal(SNR,a,mu,M,phi,D,rinit,Interpolating_Eps_Inf_Functions,r,t,delta_t,freq_bin,PSD):
    GW_pad = zero_pad(Waveform_All_Modes(r,t,a,mu,M,phi,D,Interpolating_Eps_Inf_Functions))
    Normalise = Inner_Prod(GW_pad,GW_pad,delta_t,freq_bin,PSD) # Find normalising factor.
    return (SNR/np.sqrt(Normalise))*GW_pad
#
#
#EpsFun = Extrapolate(1-10**-9)  # Extract the extrapolating function for the relativistic
#Interpolating_Eps_Inf_Functions = ExtrapolateInf_All(1-10**-9)
#SNR = 20
#a = 1-10**-9
#mu = 10
#M = 1e7
#phi = 0
#D = 1
#rinit = 1.2
#
#r,t,delta_t = Radial_Trajectory(a,mu,M,rinit,EpsFun)
#
#n = len(zero_pad(r))   # Sample size in the time domain
#fs = 1 / delta_t  # Sampling rate (measured in 1/seconds)
#nyquist = fs / 2  # Nyquist frequency 
#    
#if n % 2:  # Odd
#    n_f = (n - 1) // 2  # Note // rather than / for integer division!
#else:  # Even
#    n_f = n // 2 + 1        
#freq_bin = np.linspace(0, np.pi, n_f) * nyquist / np.pi # In units of Hz. 
#freq_bin = np.delete(freq_bin,0)        # Remove the zeroth frequency bin
#PSD = PowerSpectralDensity(freq_bin)  # Compute PSD
#
#waveform = signal(SNR,a,mu,M,phi,D,rinit,Interpolating_Eps_Inf_Functions,r,t,delta_t,freq_bin,PSD)

    
    


































    
    
    