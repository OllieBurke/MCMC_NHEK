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
warnings.filterwarnings('ignore')  # Because of stiff differential equation.

def units():
    """
    Bloody units... throughout the entire code, we define everything in terms of
    seconds. Because units are shit.
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

def R(a,mu,M,rinit):
    '''
    INPUTS: Spin a and initial radii to begin inspiral rinit
    
    OUTPUTS: inspiral radial trajectory r, Boyer Lindquist time t and sampling interval delta_t
    This function uses a standard python integrator to numerically integrate the differential equation
    above. It will return the values r against t alongside the interpolating/extrapolating 
    functions. Want to generalte inspiral from w0 = 6.714M since this will generate a year longh inspiral.    
    
    WARNING: outputs r_insp in units of M. t_insp in units of M/eta. MUST multiply by M * Msun_sec to find equivalent in seconds.
    '''

    Distance_sec,Msun_sec = units()
    EpsFun = Extrapolate(a)  # Extract the extrapolating function for the relativistic
                             # correction values.

    r_isco = risco(a,1,1)[0] # Calculate the ISCO at a spin of a
    
    
# =============================================================================
#   The below should be used when we run the algorithm properly. Reason: It
#   Calculates the sample rate at which we can resolve the highest frequency of
#   signal. For the sake of pace, we choose a smaller stepsize.
# =============================================================================
    
#    #freq_max = (11/(2*np.pi)) * ((r_isco**(3/2)+a)**-1)/4.97  # Highest frequency, units of seconds. We use 11 harmonics remember.
#    freq_max = ((11/(2*np.pi)) * (2)**-1)/(1e6 * Msun_sec)  # I already know the highest frequency since we are on a circular orbit.
                                                                # As such, I can decipher the frequency at the ISCO (units of Hz)
    
# REMEMBER: f_{m} = (1/2pi) * m * orbital velocity. m = 11 is the largest frequency... although in power it is small it is still present.    
    

#    #    print('largest frequency is',freq_max)
#    fs = 2*freq_max +0.001      # set sample rate > 2*f_max so we can resolve highest frequency (we are oversampling)
#    
#    delta_t = ((1e-5)/(1e6*Msun_sec) * (1/fs))    # Work in radiation reaction units. This itself is dimensionless.
#    # However, I must multiply by M/eta to find the value in seconds.
#    #    delta_t = 0.001232  # Test for quickness.
#    
#    delta_t_sec = delta_t * 1e6 * Msun_sec / eta
#    #rinit = 6.714          # Initial value 6.714M (boundary condition) (year long at a = 1-10**-9)
    
    eta = mu/M
    
    delta_t = 20 # Choose nice and small delta_t. This is in units of seconds. This is our sampling interval.

    t0 = 0 # initial time.

    r = ode(diffeqn, jac = None).set_integrator('lsoda', method='bdf')
    r.set_initial_value(rinit,t0).set_f_params(a,EpsFun,r_isco,eta)
    
    t_final = np.pi*1e7 # integrate until 10 M/eta has been achieved.
    
    r_insp = []
    t_insp = []
    
    j = 0    
    while r.successful() and r.t < t_final:
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
#    sky_averaging_constant = (20/3) # Sky Averaged <--- I got this from Jonathan's notes but I need
#                                    # to check where he got it...
#    L = 2.5*10**9   # Length of LISA arm
#    f0 = 19.09*10**-3    
#
#    Poms = ((1.5*10**-11)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
#    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
#    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
#                                            + np.tanh(1680*(0.00215 - f)))  
#
#    PSD = (sky_averaging_constant)* ((10/(3*L**2))*(Poms + (4*Pacc)/((2*np.pi*f))**4)*(1 + 0.6*(f/f0)**2) + Sc) # PSD
    PSD = (20/3) * 1e-40   # Sky averaged white noise 

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
    
    Interpolating_Eps_Inf_Functions,Interpolating_Eps_Inf_998_Functions = ExtrapolateInf_All(a)[0:2]
    # Extract interpolating functions
    Distance_sec,Msun_sec = units()  # Extract helpful units
    # Extract the two interpolating functions for the fluxes
    if a == 0.998:
        # Use the exact representation of the waveform.
        Interpolating_Eps_Inf_Functions = Interpolating_Eps_Inf_998_Functions 
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


def Inner_Prod(signal1,signal2,delta_t,freq_bin,PSD):
    """ 
    Computes inner product of signal
    IMPORTANT: here we use the DISCRETE fourier transform rather than the 
    continuous fourier transform. 
    See 
    
    https://www.rdocumentation.org/packages/bspec/versions/1.5/topics/snr
    
    for the formula
    """
    
    n_f = len(freq_bin)   # length of signal in frequency doamin
    n_t = len(signal1)    # length of signal in the time domain
    FFT_Sig1 =  np.fft.fft(signal1)[0:n_f]  # Calculate fft
    FFT_Sig2 = np.fft.fft(signal2)[0:n_f]   # Calculate fft
    
    Inner_Product = 4*delta_t*sum(np.real((FFT_Sig1 * np.conjugate(FFT_Sig2))) / (n_t * PSD))
    
    return Inner_Product

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
    
    FFT_GW = np.fft.fft(GW_pad)[0:n_f]  # Find FFT. ONLY using the positive frequencies.


    PSD = PowerSpectralDensity(freq_bin)  # Compute PSD
    
    Normalise = Inner_Prod(GW_pad,GW_pad,delta_t,freq_bin,PSD) # Find normalising factor.

    return Normalise,GW_pad,freq_bin,PSD,r,new_t,Interpolating_Eps_Inf_Functions,delta_t,last_index


def GW_Normed(SNR,a,mu,M,phi,D,rinit):
    '''
    Generates the normalised signal for SNR = whatever I want. 
    '''
     
    Normalise,GW_pad,freq_bin,PSD,r,new_t,Interpolating_Eps_Inf_Functions,delta_t,last_index = Un_Normed(a,mu,M,phi,D,rinit)
    # Calculate the Normalising factor
    Distance_sec,Msun_sec = units()  # Extract helpful units
    # Read in distance in seconds.
    Distance = (((SNR/np.sqrt(Normalise))**-1)) / (Distance_sec)
    
    # Calculate the distance in Gpc
    
    
    GW_pad *= (SNR / np.sqrt(Normalise))  # The signal has SNR for what I specified.
    

    return GW_pad,new_t,delta_t,freq_bin,PSD,Normalise,last_index
        
    
def Overlap(a,b,delta_t,freq_bin,PSD):
    """
    This function computes overlaps between to signals a and b
    """
    numerator = Inner_Prod(a,b,delta_t,freq_bin,PSD)
    denominator = np.sqrt((Inner_Prod(a,a,delta_t,freq_bin,PSD) * Inner_Prod(b,b,delta_t,freq_bin,PSD)))
    overlap = numerator / denominator
    return overlap
    
def Fisher_a(a,rinit):
        
    """
    1. For a large number of the spins, the overlaps are not symmetric.
    2. The uncertainties are completely ridiculous.
    3. The fisher matrix probably assumes that h follows the same parametrization
        it may not in this case. The Fisher matrix also knows nothing about the 
        upper bound on the spin parameter. 
    4. As I increase the spin, there is no obvious behaviour in how the
        uncertainty changes. For example, fixing r_init and running the code
        for spins 1-10^-i for i in (3,15) does not mean the uncertainty
        drops... I would have expeceted that the uncertainty would drop due to 
        the extra cycles gained since the ISCO --> horizon.
    5. The derivatives look extremely strange. Not stable at all as I 
        vary the perturbing parameter. Perhaps need to look for a more
        accurate way of computing the derivative. Also, very hard to take the
        derivative of a waveform when there exists an upper bound on the spin. 
        EG: Test spin a = 1-10^-12... then I can only use a perturbing value
        10^-14 and above...
    6. I personally believe it has something to do with the waveform model.
        That is... the computation of the fluxes. Jon thinks that there
        is something wrong with the way I'm computing the Fisher matrix...
        I do NOT believe it has anything to do with the Fisher matrix... but, 
        then again, I did try and convince him that I had discovered complex (valued)
        orbits in the Kerr spacetime.
    7. Limit of high SNR, overlaps should behave like 1 - 1/SNR**2 . They sometimes
        do and they sometimes don't.
        
    EDIT: New thoughts
    
    1. Question (posed by Jon): Are we able to identify near-extremal black holes
       which have a spin parameter greater than 0.998? 
    2. If, my waveform model is terrible (which I am 0.9975%) sure is the case, 
       then perhaps I could interpolate yet more flux data which I have been generating.
       Jon thinks that this should not be the problem though... I really hope he is right.
    3. Perhaps there is a rouge factor of the mass ratio floating around? 
    4. Units? Probabiy. I have tried to be as careful as possible.
    5. Maybe because GW(a1-delta_a) and GW(a1 + delta_a) have different 
       normalising constants which will cause problems? Don't know why this would
       be though.
    6. Work on a log scale?
    7. There is a massive spike right at the end of the derivative. Why?
    
    8. The reason I got "reasonable" fisher matrix estimates last time was 
        because I was only differentiating the angular frequency with
        respect to a.
        
    9. Fisher matrix on the secondary mass is also showing explosive behaviour...
    """
    

    rinit = 3                           # Specify initial radii to start
    
    print('Spin is ',a)
    mu = 10      # Secondary mass
    M = 1e6      # Primary mass
    eta = mu/M   # Mass ratio
    phi = 0      # Initial phase
    D = 1        # Distance (will) be chosen later to give SNR specified
    SNR = 20     # SNR of signal
    deriv_perturb = 1e-10   # Perturbing value to compute derivative
# =============================================================================
#     Calcaulte derivative of waveform with respect to a
# =============================================================================
    a1 = a + deriv_perturb  
    a2 = a - deriv_perturb
    
    GW_pad1,new_t,delta_t,freq_bin,PSD,Normalise,last_index1 = GW_Normed(SNR,a1,mu,M,phi,D,rinit)
    GW_pad2,new_t,delta_t,freq_bin,PSD,Normalise,last_index2 = GW_Normed(SNR,a2,mu,M,phi,D,rinit)
    
    
    diff = ((GW_pad1 - GW_pad2))/(2*deriv_perturb)
    deriv = high_order_deriv(SNR,a,mu,M,phi,D,rinit)  # Here we use a higher 
                                                      # order derivative
# =============================================================================
#       Plot the derivative    
# =============================================================================
    plt.plot(new_t,diff)
    plt.xlabel(r'$\tilde{t}/\eta$')
    plt.ylabel(r'$\tilde{r}$')
    plt.title(r'Derivative of Waveform wrt $a$')
    plt.show()
    plt.clf()
    
    plt.plot(new_t,deriv)
    plt.xlabel(r'$\tilde{t}/\eta$')
    plt.ylabel(r'$\tilde{r}$')
    plt.title(r'Derivative of Waveform wrt $a$')
    plt.show()
    plt.clf()
    
    # Compute 1-sigma deviation in likelihood from MLE.
    uncertainty_a = np.sqrt(Inner_Prod(diff,diff,delta_t,freq_bin,PSD)**-1)   
# =============================================================================
#     Below we compute overlaps of signals where we perturb the spin by
#     the uncertainty \Delta a. The results should be symmetric and approximately
#     overlap = 1 - 1/\rho^{2}
# =============================================================================
    
    a1 = a 
    a2 = a - uncertainty_a
    a3 = a + uncertainty_a
    GW_pad1,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a1,mu,M,phi,D,rinit)
    GW_pad2,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a2,mu,M,phi,D,rinit)
    GW_pad3,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a3,mu,M,phi,D,rinit)
    
    
    overlap_plus_minus = Overlap(GW_pad2,GW_pad3,delta_t,freq_bin,PSD)
    overlap_minus = Overlap(GW_pad1,GW_pad2,delta_t,freq_bin,PSD)
    overlap_plus = Overlap(GW_pad1,GW_pad3,delta_t,freq_bin,PSD)
    
# =============================================================================
#     Output
# =============================================================================
    print('spin is',a)
    print('Uncertainty in is',uncertainty_a) 
    print('The overlap between the two waveforms (a,a-delta_a) using fisher result is', overlap_minus)
    print('The overlap between the two waveforms (a,a+delta_a) using fisher result is', overlap_plus)
    print('The overlap between the two waveforms (a-delta_a,a+delta_a) using fisher result is', overlap_plus_minus)
    print('\n We should be getting an overlap of ',(1 - SNR**-2 + (1/4)*SNR**-4))


def Overlap_GWs(delta_a):
    GW_pad,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(20,0.998+delta_a,10,1e6,0,1,3)
    GW_thorne,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(20,0.998,10,1e6,0,1,3)  

    overlap = Overlap(GW_pad,GW_thorne,delta_t,freq_bin,PSD)
    print(overlap)
    print('Overlap of spin a = %s and a = 1-10**-9 is %s'%(a,overlap))
    
def high_order_deriv(SNR,a,mu,M,phi,D,rinit):
    """
    This is a more accurate way to compute the derivative (although slower).
    I am getting the same derivative as I was when using the formula for
    symmetric differences. I get the same Fisher matrix result using this formula.
    """
    deriv_perturb = 1e-10
    
    GW_pad_plus2,new_t,delta_t,freq_bin,PSD,Normalise,last_index1 = GW_Normed(SNR,a+2*deriv_perturb,mu,M,phi,D,rinit)
    GW_pad_plus1,new_t,delta_t,freq_bin,PSD,Normalise,last_index1 = GW_Normed(SNR,a+deriv_perturb,mu,M,phi,D,rinit)    
   
    GW_pad_minus1,new_t,delta_t,freq_bin,PSD,Normalise,last_index1 = GW_Normed(SNR,a-deriv_perturb,mu,M,phi,D,rinit)    
    GW_pad_minus2,new_t,delta_t,freq_bin,PSD,Normalise,last_index1 = GW_Normed(SNR,a-2*deriv_perturb,mu,M,phi,D,rinit)
    
    deriv = ((-GW_pad_plus2 + 8*GW_pad_plus1 - 8*GW_pad_minus1 + GW_pad_minus2))/(12*deriv_perturb)
    return deriv


def Fisher_a_tukey(a,rinit):
        
    """
    To test whether the problem could be coming from a hard cutoff I will try
    and window the signal. Perhaps the extra certainty could come from 
    extreme spectral leakage since we are windowing with a rectangular function    
    """

    a = 0.998  # Spin of choice <-- I have exact flux data for this.
    rinit = 3  # initial radii
    Distance_sec,Msun_sec = units()
    
# =============================================================================
#     Parameters of the system.
# =============================================================================
    mu = 10
    M = 1e6
    eta = mu/M
    phi = 0
    D = 1
    SNR = 20
    deriv_perturb = 1e-10
# =============================================================================
# Calculate derivative    
# =============================================================================
    a1 = a + deriv_perturb
    a2 = a - deriv_perturb
    
    Interpolating_Eps_Inf_Functions1,Interpolating_Eps_Inf_998_Functions = ExtrapolateInf_All(a1)[0:2]
    Interpolating_Eps_Inf_Functions2,Interpolating_Eps_Inf_998_Functions = ExtrapolateInf_All(a2)[0:2]
    
    GW_pad1,new_t,delta_t,freq_bin,PSD,Normalise1,last_index1 = GW_Normed(SNR,a1,mu,M,phi,D,rinit)
    GW_pad2,new_t,delta_t,freq_bin,PSD,Normalise2,last_index2 = GW_Normed(SNR,a2,mu,M,phi,D,rinit)
    
    r_isco_a1 = risco(a1,1,1)[0]
    r_isco_a2 = risco(a2,1,1)[0]

# =============================================================================
# both signal_end_1 and signal_end_2 will be windowed shortly. I have chosen
# the value of r_isco since this will specify the final frequency of the 
# signal. This makes sense in my head...    
# =============================================================================
    signal_end_1 = (SNR / np.sqrt(Normalise1)) * Waveform_All_Modes(r_isco_a1,new_t[last_index1:],a1,mu,M,phi,D,Interpolating_Eps_Inf_Functions1)
    signal_end_2 = (SNR / np.sqrt(Normalise1)) * Waveform_All_Modes(r_isco_a2,new_t[last_index2:],a2,mu,M,phi,D,Interpolating_Eps_Inf_Functions2)
    
    alpha = 0.2 # Windowing parameter, specifies length of cosine lobes. 
# =============================================================================
#     Here we build our own tukey window. One sided and, more specifically,
#     a reverse tukey window (1 - window).
# =============================================================================
    window1 = []
    N_1 = len(signal_end_1)
    for i in range(0,N_1):
        if i < np.floor(alpha * N_1/2):  # for indices less than the length of the signal
            window1.append(0.5*(1 + np.cos(np.pi*(2*i/(alpha*N_1) - 1)))) # form window: one sided turkey window
        else:
            window1.append(1) # After cosine lobe the tukey window = 1.
            
    window2 = []
    N_2 = len(signal_end_2)
    for i in range(0,N_2):
        if i < np.floor(alpha * N_2/2):
            window2.append(0.5*(1 + np.cos(np.pi*(2*i/(alpha*N_2) - 1))))  # One sided turkey window
        else:
            window2.append(1)
            
    reverse_tukey1 = 1 - np.array(window1)  # Reverse turkey
    reverse_tukey2 = 1 - np.array(window2)  # Reverse turkey
            
    length_tukey1 = len(signal_end_1)
    length_tukey2 = len(signal_end_2)   # Compute lengths.
    
# =============================================================================
#     Window both the signals below
# =============================================================================
    
    signal_full_1 = np.concatenate([GW_pad1[0:last_index1],signal_end_1*reverse_tukey1])
    signal_full_2 = np.concatenate([GW_pad2[0:last_index2],signal_end_2*reverse_tukey2])
    
       
    diff = ((signal_full_1 - signal_full_2))/(2*deriv_perturb) # Compute derivative
    
    plt.plot(new_t,diff) 
    plt.xlabel(r'$\tilde{t}/\eta$')
    plt.ylabel(r'$\tilde{r}$')
    plt.title(r'Derivative of Waveform wrt $a$')
    plt.show()
    plt.clf()
    
# =============================================================================
#     Compare the effects of windowing by computing uncertainties below.
# =============================================================================
    
    uncertainty_a = np.sqrt(Inner_Prod(deriv,deriv,delta_t,freq_bin,PSD)**-1)   
    uncertainty_a_windowing = np.sqrt(Inner_Prod(diff,diff,delta_t1,freq_bin,PSD)**-1)
    
    print('without windowing',uncertainty_a)
    print('with windowing',uncertainty_a_windowing)

def Fisher_mu(a,rinit):
    """
    Here we compute a 1 dimensional fisher matrix on the secondary mass mu.
    We highlight here that the results we obtain seem sensible and are verified
    using the formula for the overlaps (1 - 1/\rho^2). This highlights to me
    that the problem with the waveform model is when we are changing the spin
    which directly influences the fluxes. As a result, the spin is screwing us.
    
    The details below are essentially identical to the function 
    Fisher_a(a,rinit). So, it is a waste of time commenting it. See the function
    above.
    """   

    
    print('Spin is ',a)
    mu = 10
    M = 1e6
    eta = mu/M
    phi = 0
    D = 1
    SNR = 20
    deriv_perturb = 1e-8

    mu1 = mu + deriv_perturb
    mu2 = mu - deriv_perturb
    
    GW_pad1,new_t,delta_t,freq_bin,PSD,Normalise,last_index1 = GW_Normed(SNR,a,mu1,M,phi,D,rinit)
    GW_pad2,new_t,delta_t,freq_bin,PSD,Normalise,last_index2 = GW_Normed(SNR,a,mu2,M,phi,D,rinit)
    
    
    diff = ((GW_pad1 - GW_pad2))/(2*deriv_perturb)
    
    plt.plot(new_t,diff)
    plt.xlabel(r'$\tilde{t}/\eta$')
    plt.ylabel(r'$\tilde{r}$')
    plt.title(r'Derivative of Waveform wrt $a$')
    plt.show()
    plt.clf()
    
    uncertainty_mu = np.sqrt(Inner_Prod(diff,diff,delta_t,freq_bin,PSD)**-1)   
    
    print(uncertainty_mu)
    
    mu1 = mu 
    mu2 = mu - uncertainty_mu
    mu3 = mu + uncertainty_mu
    GW_pad1,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a,mu1,M,phi,D,rinit)
    GW_pad2,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a,mu2,M,phi,D,rinit)
    GW_pad3,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a,mu3,M,phi,D,rinit)
    
    
    plt.plot(new_t,GW_pad1)
    plt.xlabel(r'$\tilde{t}/\eta$')
    plt.ylabel(r'$\tilde{r}$')
    plt.title(r'Waveform with spin $a$')
    plt.show()
    plt.clf()
    
    overlap_plus_minus = Overlap(GW_pad2,GW_pad3,delta_t,freq_bin,PSD)
    overlap_minus = Overlap(GW_pad1,GW_pad2,delta_t,freq_bin,PSD)
    overlap_plus = Overlap(GW_pad1,GW_pad3,delta_t,freq_bin,PSD)
    
    print('secondary mass is',mu)
    print('Uncertainty in is',uncertainty_mu) 
    print('The overlap between the two waveforms (a,a-delta_a) using fisher result is', overlap_minus)
    print('The overlap between the two waveforms (a,a+delta_a) using fisher result is', overlap_plus)
    print('The overlap between the two waveforms (a-delta_a,a+delta_a) using fisher result is', overlap_plus_minus)
    print('\n We should be getting an overlap of ',(1 - SNR**-2 + (1/4)*SNR**-4))
    
def Fisher_M():
    """
    Again, I am getting very sensible results for uncertainty of M which leads
    me to believe that I am experiencing a problem with with the spin parameter.
    
    No need to comment this since it is much the same as the other functions
    above.
    """
        
    
    a = 0.998
    rinit = 3
    Distance_sec,Msun_sec = units()
    
    print('Spin is ',a)
    mu = 10
    M = 1e6
    eta = mu/M
    phi = 0
    D = 1
    SNR = 20
    deriv_perturb = 1e-2
    
    M1 = M + deriv_perturb
    M2 = M - deriv_perturb
    
    GW_pad1,new_t,delta_t,freq_bin,PSD,Normalise,last_index1 = GW_Normed(SNR,a,mu1,M1,phi,D,rinit)
    GW_pad2,new_t,delta_t,freq_bin,PSD,Normalise,last_index2 = GW_Normed(SNR,a,mu2,M2,phi,D,rinit)
    
    
    diff = ((GW_pad1 - GW_pad2))/(2*deriv_perturb)
    
    plt.plot(new_t,diff)
    plt.xlabel(r'$\tilde{t}/\eta$')
    plt.ylabel(r'$\tilde{r}$')
    plt.title(r'Derivative of Waveform wrt $a$')
    plt.show()
    plt.clf()
    
    uncertainty_M = np.sqrt(Inner_Prod(diff,diff,delta_t,freq_bin,PSD)**-1)   
    
    print(uncertainty_M)
    
    M1 = M
    M2 = M - uncertainty_M
    M3 = M + uncertainty_M
    GW_pad1,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a,mu,M1,phi,D,rinit)
    GW_pad2,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a,mu,M2,phi,D,rinit)
    GW_pad3,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a,mu,M3,phi,D,rinit)
    
    
    overlap_plus_minus = Overlap(GW_pad2,GW_pad3,delta_t,freq_bin,PSD)
    overlap_minus = Overlap(GW_pad1,GW_pad2,delta_t,freq_bin,PSD)
    overlap_plus = Overlap(GW_pad1,GW_pad3,delta_t,freq_bin,PSD)
    
    print('secondary mass is',M)
    print('Uncertainty in is',uncertainty_M) 
    print('The overlap between the two waveforms (a,a-delta_a) using fisher result is', overlap_minus)
    print('The overlap between the two waveforms (a,a+delta_a) using fisher result is', overlap_plus)
    print('The overlap between the two waveforms (a-delta_a,a+delta_a) using fisher result is', overlap_plus_minus)
    print('\n We should be getting an overlap of ',(1 - SNR**-2 + (1/4)*SNR**-4))

def deriv_GRC(a):
    """
    This function computes the derivative of the general relativistic 
    correction with respect to a. I am finding some really funny results here.
    Sometimes the derivative of the function blows up and other times it's 
    perfectly fine.
    
    """
    
    deriv_perturb = 1e-12
    
    a1 = a + deriv_perturb
    a2 = a - deriv_perturb
    
    Interpolating_Eps_Inf_Functions1,Interpolating_Eps_Inf_998_Functions = ExtrapolateInf_All(a1)[0:2]
    Interpolating_Eps_Inf_Functions2,Interpolating_Eps_Inf_998_Functions = ExtrapolateInf_All(a2)[0:2]
    
    # The above extracts the interpolating functions for the flux data for 
    # each of the spins a1 and a2.
    
    Epsfun1 = Interpolating_Eps_Inf_Functions1[0]  # Here we use the m = 2 flux interpolant
    Epsfun2 = Interpolating_Eps_Inf_Functions2[0]  # Similar.
        
    r_isco1 = risco(a1,1,1)[0]  # ISCO for spin a1
    r_isco2 = risco(a2,1,1)[0]  # ISCO for spin a2
    
    r1 = np.linspace(r_isco1,1.25,10000) # Create radial coordinates for interpolant
    r2 = np.linspace(r_isco2,1.25,10000) # Same.
    
    EpsVal1 = Epsfun1(r1);EpsVal2 = Epsfun2(r2) # Compute values of interpolant at 
                                                # each r1 and r2.
    
    deriv = (EpsVal1 - EpsVal2)/(2*deriv_perturb) # Compute derivative
# =============================================================================
#     Plot the results.
# =============================================================================
    plt.plot(r1,deriv)
    plt.title('derivative')
    plt.show()
    plt.clf()

    plt.plot(r1,EpsVal1,'r--',label = 'spin+')    
    plt.plot(r2,EpsVal2,'k',label = 'spin-')
    plt.legend()
    plt.title('spin+')
    plt.show()
    plt.clf()
    
    plt.plot(r2,np.log10(EpsVal2-EpsVal1))
    plt.title('log plot')
    plt.ylabel(r'log of difference')
    plt.xlabel(r'$r$')
    

       
def Overlap_primary_mass(M):
    """
    My MCMC results are screwing up on the primary mass... here we can 
    investigate it a little further by computing overlaps of the two signals.
    """
# =============================================================================
#     Parameters
# =============================================================================
    a = 0.998
    mu = 10
    phi = 3
    D = 1
    M_exact = 1e6
    SNR = 20
    rinit = 1.32
# =============================================================================
#     Extract two signals.
# =============================================================================
    GW_M,new_t_exact,delta_t,freq_bin,PSD,Normalise,last_index1 = GW_Normed(SNR,a,mu,M_exact,phi,D,rinit)
    GW_M_try,new_t_try,delta_t,freq_bin,PSD,Normalise,last_index1 = GW_Normed(SNR,a,mu,M,phi,D,rinit)
    
    """
    
    In the code, where $M$ appears is simply in the mass ratio (since we) 
    work in seconds and $r$ is simply rescaled with the total mass M.
    
    As a result, by increasing $M$, eta is made smaller which implies there 
    are more cycles as a result. Since the time of inspiral scales as 
    1/eta, the lengths of the (non padded) signal will change.

    We need to be very careful about this. Below is an attempt to 
    make the signals the same length without destroying any valuable
    information in the signal.
    
    """
    if len(GW_M) < len(GW_M_try):  
        GW_M_try = GW_M_try[0:len(GW_M)] # Truncate new signal GW_M_try so it is
                                         # the same length as GM_M.
        new_t_try = new_t_try[0:len(GW_M)] # truncate the times.
    else:
        GW_M = GW_M[0:len(GW_M_try)]  # Truncate GW_M so it is of the same length
                                      # as GW_M_try
        new_t_exact = new_t_exact[0:len(GW_M_try)] # truncate the times
        
        
    overlap = Overlap(GW_M,GW_M_try,delta_t,freq_bin,PSD) # Compute overlap.
# =============================================================================
#     Plot
# =============================================================================
    plt.plot(new_t_exact,GW_M_try,'darkviolet')
    plt.show()
    plt.clf()
    plt.plot(new_t_try,GW_M,'blue')
    print('Overlap is',overlap)
    
    
    
    
    
    
    
    