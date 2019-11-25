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
import matplotlib

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
    

    #rinit = 6.714          # Initial value 6.714M (boundary condition) (year long at a = 1-10**-9)
    
    eta = mu/M
    
    delta_t = 1 # Choose nice and small delta_t. This is in units of seconds. This is our sampling interval.

    t0 = 0 # initial time.

    r = ode(diffeqn, jac = None).set_integrator('lsoda', method='bdf')
    r.set_initial_value(rinit,t0).set_f_params(a,EpsFun,r_isco,eta)
#    one_day = 86400

    t_final = 2*np.pi*1e7
    
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
    
    Interpolating_Eps_Inf_Functions,Interpolating_Eps_Inf_998_Functions = ExtrapolateInf_All(a)[0:2]
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
#    SNR = 20
#    a = 1-10**-9
#    mu = 10
#    M = 1e7
#    phi = 0
#    D = 1
#    rinit = 5.379  # 1 year with eta = 10^{-6}, M = 1e7 <-- better for NHEK.
#    print('initial radii',rinit)
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
        
    























































def Overlap(a,b,delta_t,freq_bin,PSD):
    """
    This function computes overlaps between to signals a and b
    """
    numerator = Inner_Prod(a,b,delta_t,freq_bin,PSD)
    denominator = np.sqrt((Inner_Prod(a,a,delta_t,freq_bin,PSD) * Inner_Prod(b,b,delta_t,freq_bin,PSD)))
    overlap = numerator / denominator
    return overlap
     
































# =============================================================================
# SNR Calculations
# =============================================================================
def SNR_NHEK(SNR,a,mu,M,phi,D,rinit):
    GW_pad,new_t,delta_t,freq_bin,PSD,Normalise,last_index = GW_Normed(SNR,a,mu,M,phi,D,rinit)
    
    max_index = np.argmax(GW_pad)
    GW_NHEK = GW_pad[max_index:]
    
    n = len(GW_NHEK)   # Sample size in the time domain
    fs = 1 / delta_t  # Sampling rate (measured in 1/seconds)
    nyquist = fs / 2  # Nyquist frequency 
        
    if n % 2:  # Odd
        n_f = (n - 1) // 2  # Note // rather than / for integer division!
    else:  # Even
        n_f = n // 2 + 1        
    freq_bin = np.linspace(0, np.pi, n_f) * nyquist / np.pi # In units of Hz. 
    freq_NHEK = np.delete(freq_bin,0)        # Remove the zeroth frequency bin
    PSD = PowerSpectralDensity(freq_NHEK)
    
    SNR2 = Inner_Prod(GW_NHEK,GW_NHEK,delta_t,freq_NHEK,PSD)
# =============================================================================
# Cool plots
# =============================================================================
    
def R_plot():
    eta = 1e-5
    
    r_a9,t_a9,delta_t = R(1-10**-9,10,1e6,3)
    r_a6,t_a6,delta_t = R(1-10**-6,10,1e6,3)
    r_a3,t_a3,delta_t = R(1-10**-3,10,1e6,3)
    
    plt.plot(t_a9*1e-5,r_a9,label = r'$a=1-10^{-9}$')
    plt.plot(t_a6*1e-5,r_a6,label = r'$a=1-10^{-6}$')
    plt.plot(t_a3*1e-5,r_a3,label = r'$a=1-10^{-3}$')

    plt.xlabel(r'$\tilde{t}\times \eta$')
    plt.ylabel(r'$\tilde{r}$')
    plt.title('Radial Evolution')        
    plt.legend(prop = {'size':18})
    
def f(r,a):
    return -1*(64/5)*((r)**(3/2) + a)**(-10/3)*  ((1 - (3/r) + 2*a*((1/r)**(3/2)))**(3/2))/ ((1-(6/r) + 8*a*(1/r)**(3/2) - 3*(a/r)**2)*((1/r)**2))

def inspiral(a):
    r_isco = risco(a,1,1)[0]
    EpsFun = Extrapolate(a)
    
    drdt = []
    r0 = np.arange(r_isco+0.00001,8,0.00001)
    return EpsFun(r0)*f(r0,a),r0
    
def plot_deriv():
    #    drdt,r0 = inspiral(1-10**-13)
    #    drdt_13 = drdt
    #    r0_13 = r0
    #    drdt,r0 = inspiral(1-10**-12)
    #    drdt_12 = drdt
    #    r0_12 = r0
    #    drdt,r0 = inspiral(1-10**-11)
    #    drdt_11 = drdt
    #    r0_11 = r0
    #    drdt,r0 = inspiral(1-10**-10)
    #    drdt_10 = drdt
    #    r0_10 = r0
    #    drdt,r0 = inspiral(1-10**-9)
    #    drdt_9 = drdt
    #    r0_9 = r0
    drdt,r0 = inspiral(1-10**-8)
    drdt_8 = drdt
    r0_8 = r0
    drdt,r0 = inspiral(1-10**-7)
    drdt_7 = drdt
    r0_7 = r0
    drdt,r0 = inspiral(1-10**-6)
    drdt_6 = drdt
    r0_6 = r0
    drdt,r0 = inspiral(1-10**-5)
    drdt_5 = drdt
    r0_5 = r0
    drdt,r0 = inspiral(1-10**-4)
    drdt_4 = drdt
    r0_4 = r0
    drdt,r0 = inspiral(1-10**-3)
    drdt_3 = drdt
    r0_3 = r0
    drdt,r0 = inspiral(0.998)
    drdt_2 = drdt
    r0_2 = r0
    drdt,r0 = inspiral(0.98)
    drdt_1 = drdt
    r0_1 = r0
    
    #    plt.plot(r0_13,drdt_13)
    #    plt.plot(r0_12,drdt_12)
    #    plt.plot(r0_11,drdt_11)
    #    plt.plot(r0_10,drdt_10)
    #    plt.plot(r0_9,drdt_9)
    #    plt.plot(r0_8,drdt_8)
    #    plt.plot(r0_7,drdt_7)
    plt.plot(r0_8,drdt_8,label = r'$a = 1-10^{-8}$')    
    plt.plot(r0_7,drdt_7,label = r'$a = 1-10^{-7}$')
    plt.plot(r0_6,drdt_6,label = r'$a =1-10^{-6}$')
    plt.plot(r0_5,drdt_5,label = r'$a =1-10^{-5}$')
    plt.plot(r0_4,drdt_4,label = r'$a =1-10^{-4}$')
    plt.plot(r0_3,drdt_3,label = r'$a=1-10^{-3}$')

    plt.xlim([0.8,3.5])
    plt.ylim([-0.8,0]) 
    plt.title('Radial Derivative', size = 18)
    plt.xlabel(r'$\tilde{r}$',size=18)
    plt.ylabel(r'$d\tilde{r}/d\tilde{t}$',size = 18)
    plt.legend(prop = {'size': 18})
    
def Time_Until_ISCO(a):
    Distance_sec,Msun_sec = units()
    EpsFun = Extrapolate(a)
    #r_true,t_true = R(a,h)
    r_isco = risco(a,1,1)[0]
#    r_horiz = 1 + np.sqrt(1-a**2)
    r = np.arange(1.473,r_isco,-0.00001)

    N_Cor = (1/ EpsFun(r)) * (1 + a/(r**(3/2)))**(5/3) * (1 - 6/r + 8*a/(r**(3/2)) - 3*(a/r)**2) * ((1 - 3/r + 2*a/(r**(3/2)))**(-3/2))

    limits = np.sqrt(Omega(r,a,0))

    T_Cor = (8/3) * Omega(r[0],a,1) * simps(N_Cor/ Omega(r,a,5/2) ,limits) # looks ok.

    T = ((5/256)*(1e6 * Msun_sec/1e-5) / (Omega(r[0],a,1))) * T_Cor
    
    t_rad_reaction = T /60/60/24


    return t_rad_reaction
    
def No_Orbits(a):
    EpsFun = Extrapolate(a)
    r_isco = risco(a,1,1)[0]
    r = np.arange(1.473,r_isco,-0.0001)
    #r = np.arange(6.714,1.473,-0.0001)

    N_Cor = (1/ EpsFun(r)) * (1 + a/(r**(3/2)))**(5/3) * (1 - 6/r + 8*a/(r**(3/2)) - 3*(a/r)**2) * ((1 - 3/r + 2*a/(r**(3/2)))**(-3/2))

    limits = np.sqrt(Omega(r,a,0))
    
    N_Orb_Cor = (5/3) * Omega(r[0],a,-1/2) *simps(N_Cor/Omega(r,a,1),limits)
    N_orbits = ((1/(64*np.pi)) * (1e5) / Omega(r[0],a,-1/2)) * N_Orb_Cor
    return N_orbits

def Info_NHEK():
    a_vec = []
    N_vec = []
    t_rad_vec = []
    
    for i in range(3,12):
        print('Spin parameter 1-10^- %s'%(i))
        a = 1-10**(-i)
        a_vec.append(i)
        
        t_rad_reaction = Time_Until_ISCO(a)
        t_rad_vec.append(t_rad_reaction)
        
        N_orbits = No_Orbits(a)
        N_vec.append(N_orbits)
        print('for a spin of %s, time in NHEK is %s days, NO. orbits %s'%(a,round(t_rad_reaction,4),np.floor(N_orbits)))
    return a_vec,t_rad_vec,N_vec

def Plot_Time_Orbits():

    
    a_vec,t_rad_vec,N_vec = Info_NHEK()
    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel(r'$a = 1-10^{-x}$')
    ax1.set_ylabel(r'$N_{Orb}$', color=color)
    ax1.set_title('Time and Number of Orbits')
    ax1.plot(a_vec, N_vec, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel(r'$T$ days ', color=color)  # we already handled the x-label with ax1
    ax2.plot(a_vec, t_rad_vec, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

#def diffeqn_back(t,r0,a,EpsFun,r_isco,eta):
#    '''
#    This is the differential equation dr0/dt = f(r0) that we wish to numerically integrate.
#    '''
#    if r0 == -inf or r0 == inf:
#
#        return nan
#    
#    elif r0<r_isco:
#        return nan
#    else:       
#        return 1*eta*EpsFun(r0)*(64/5)*((r0)**(3/2) + a)**(-10/3)*  ((1 - (3/r0) + 2*a*((1/r0)**(3/2)))**(3/2))/ ((1-(6/r0) + 8*a*(1/r0)**(3/2) - 3*(a/r0)**2)*((1/r0)**2))  
#
##def R_back(a,mu,M):
#'''
#INPUTS: Spin a and initial radii to begin inspiral rinit
#
#OUTPUTS: inspiral radial trajectory r, Boyer Lindquist time t and sampling interval delta_t
#This function uses a standard python integrator to numerically integrate the differential equation
#above. It will return the values r against t alongside the interpolating/extrapolating 
#functions. Want to generalte inspiral from w0 = 6.714M since this will generate a year longh inspiral.    
#
#WARNING: outputs r_insp in units of M. t_insp in units of M/eta. MUST multiply by M * Msun_sec to find equivalent in seconds.
#'''
#a = 1-10**-9
#mu = 10
#M = 1e6
#Distance_sec,Msun_sec = units()
#EpsFun = Extrapolate(a)  # Extract the extrapolating function for the relativistic
#                         # correction values.
#
#r_isco = risco(a,1,1)[0] # Calculate the ISCO at a spin of a
#
#
##rinit = 6.714          # Initial value 6.714M (boundary condition) (year long at a = 1-10**-9)
#
#eta = mu/M
#
#delta_t = 100 # Choose nice and small delta_t. This is in units of seconds. This is our sampling interval.
#
#t0 = 0 # initial time.
#rinit = r_isco+1e-10
#r = ode(diffeqn_back, jac = None).set_integrator('lsoda', method='bdf')
#r.set_initial_value(rinit,t0).set_f_params(a,EpsFun,r_isco,eta)
##    one_day = 86400
#
#t_final = 31536000 # seconds in a year. 
#
#r_insp = []
#t_insp = []
#
#j = 10    
#while r.t < t_final:
#    # Integrate the ODE in using a while look. The second condition is a criteria to stop.
##    print(r.t+dt, r.integrate(r.t+dt))< time step and solution r(t+dt).
#    r_int = r.integrate(r.t+delta_t)[0]   # compute each integrated r value.
#    if isnan(r_int) == True:
#        r_int = r_isco + 10**-j
#        j = j - 1
#    t_insp.append(r.t+delta_t)  # Append timestep to a list.
#    r_insp.append(r_int)
##    print('Last radial coordinate is',r_insp[-1])
##    delta_t*=(1e6 * Msun_sec/eta)
##    t_insp *= (1e6 *Msun_sec/eta)  
##    return np.array(r_insp),np.array(t_insp),delta_t    
#r_insp.sort(reverse = True)
#plt.plot(t_insp,r_insp)    
#    
    
    
    
    