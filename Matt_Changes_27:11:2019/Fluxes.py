#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:37:37 2019

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
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 

matplotlib.rcParams.update({'font.size': 14})



"""
These are the changes to my flux data. New Branch.
"""

def risco(a):
    '''
    This function computes the inner most stable circular orbit (ISCO)
    given the mass M of the black hole and its corresponding spin a.
    '''
    Z1 = 1. + ((1.-a**2)**(1/3))*((1.+a)**(1/3) + (1.-a)**(1/3))
    Z2 = (3.*a**2 + Z1**2)**(1/2)
    
    if a >= 0:       # Prograde (co-rotating) 
        r_isco = (3. + Z2 - ((3. - Z1)*(3.+Z1 + 2.*Z2))**(1/2))  # This evaluates the r_isco.
    else:            # Retrograde
        r_isco = (3. + Z2 + ((3. - Z1)*(3.+Z1 + 2.*Z2))**(1/2))
    return r_isco   # In later functions, r_isco, a and x_isco is used.

def Omega(r,a,m):
    # Particles dimensionless angular frequency
    return ((r**(3/2) + a)**-1)**(2 + 2*m/3)

def Paths():
    """
    Underneath are the hardcoded paths. You simply need to change these to the
    working directories of where the files a0.99999999, a0.998,a0.97. 
    """
    path_a09995 = '/Users/Ollie/Nextcloud/Thesis/NHEK_Geometry/NHEK_Code/MCMC_NHEK/Flux_Data/a09995'
    path_a4 = '/Users/Ollie/Nextcloud/Thesis/NHEK_Geometry/NHEK_Code/MCMC_NHEK/Flux_Data/a4' 
    path_a5 = '/Users/Ollie/Nextcloud/Thesis/NHEK_Geometry/NHEK_Code/MCMC_NHEK/Flux_Data/a5' 
    path_Ext = '/Users/Ollie/Nextcloud/Thesis/NHEK_Geometry/NHEK_Code/MCMC_NHEK/Flux_Data/a0.999999999'
    path_a0998 = '/Users/Ollie/Nextcloud/Thesis/NHEK_Geometry/NHEK_Code/MCMC_NHEK/Flux_Data/a0.998'
    path_a097 = '/Users/Ollie/Nextcloud/Thesis/NHEK_Geometry/NHEK_Code/MCMC_NHEK/Flux_Data/a0.97'
    Home_Of_Code = '/Users/Ollie/Nextcloud/Thesis/NHEK_Geometry/NHEK_Code/MCMC_NHEK'
    
    # Jonathan, put the location of each of your directories here:

#    path_Ext = '/home/mattedwards/Ollie/Gargantua_data/a0.999999999'
#    path_a0998 = '/home/mattedwards/Ollie/Gargantua_data/a0.998'
#    path_a097 = '/home/mattedwards/Ollie/Gargantua_data/a0.97'
#    Home_Of_Code = '/home/mattedwards/Ollie/'

    
    return path_a09995,path_a4,path_a5,path_Ext,path_a0998,path_a097,Home_Of_Code 

def Flux_Data():
    """
    This function reads in each of the flux data that was given to me by Niels
    back in the day. Hello there. 
    """
    path_a09995,path_a4,path_a5,path_Ext,path_a0998,path_a097,Home_Of_Code  = Paths()

    os.chdir(Home_Of_Code)
    import Flux_Combo
    Flux_Info_a0999999999 = ReadAllData_a0999999999()
    Flux_Info_a0998 = ReadAllData_a0998()
    Flux_Info_a097 = ReadAllData_a097()   # Read in all of the data flux data. This is the same for above.
    Flux_Info_a4 = ReadAllData_a4()   # Read in all of the data flux data. This is the same for above.
    Flux_Info_a09995 = ReadAllData_a09995()   # Read in all of the data flux data. This is the same for above.
    Flux_Info_a5 = ReadAllData_a5()   # Read in all of the data flux data. This is the same for above.  
    #print(len(Flux_Info_a0999999999[0][1]))
    r_999999999,Flux_999999999_Inf = [],[]
    
    for i in range(0,len(Flux_Info_a0999999999)):
        r_999999999.append(Flux_Info_a0999999999[i][0])  # Extract radial coordiantes
        Flux_999999999_Inf.append(Flux_Info_a0999999999[i][1]) # Extract (total) flux at inifinty at each radial coordiante.
        
    r_998,Flux_0998_Inf = [],[]
    for i in range(0,len(Flux_Info_a0998)):
        """
        Similar madness to the above.
        """
        r_998.append(Flux_Info_a0998[i][0])
        Flux_0998_Inf.append(Flux_Info_a0998[i][1])
    
    r_97,Flux_097_Inf = [],[]
    for i in range(0,len(Flux_Info_a097)):
        r_97.append(Flux_Info_a097[i][0])
        Flux_097_Inf.append(Flux_Info_a097[i][1])
        
    
    return Flux_999999999_Inf, r_999999999, Flux_0998_Inf, r_998, Flux_097_Inf, r_97


def EpsRelationship():
    '''
    This function will create the number of data points required to 
    interpolate the function using Pythons in built interpolating/extrapolating
    function interp1d. It essentially is built off the previous function 
    FixedSpinPlot() to create a number of data points of \Eps against r0
    '''
    
    path_a09995,path_a4,path_a5,path_Ext,path_a0998,path_a097,Home_Of_Code  = Paths()   
    
    os.chdir(Home_Of_Code)

    import Flux_Combo
    Flux_Info_a0999999999 = ReadAllData_a0999999999()
    Flux_Info_a0998 = ReadAllData_a0998()
    Flux_Info_a097 = ReadAllData_a097()   # Read in all of the data flux data. This is the same for above.
    Flux_Info_a4 = ReadAllData_a4()   # Read in all of the data flux data. This is the same for above.
    Flux_Info_a09995 = ReadAllData_a09995()   # Read in all of the data flux data. This is the same for above.
    Flux_Info_a5 = ReadAllData_a5()   # Read in all of the data flux data. This is the same for above.  

    
    r_999999999,Flux_999999999_GW = [],[]
    for i in range(0,len(Flux_Info_a0999999999)):
        r_999999999.append(Flux_Info_a0999999999[i][0])  # Radial coordinates
        Flux_999999999_GW.append(sum(Flux_Info_a0999999999[i][1]) + sum(Flux_Info_a0999999999[i][2]))  # Total energy flux
    
    Eps_999999999 = [(5/32) * 1 * ((r_999999999[i])**(3/2) + (1-10**-9))**(10/3)*Flux_999999999_GW[i] for i in range(0,len(Flux_Info_a0999999999))]
    # Calculate relativistic correction values. I do note that this is pretty dumb since I could just interpolate \dot{E}_{GW}... I'm a little smarter
    # now than I was before... only a little though.
    
    r_998,Flux_0998_GW = [],[]
    for i in range(0,len(Flux_Info_a0998)):
        r_998.append(Flux_Info_a0998[i][0])
        Flux_0998_GW.append(sum(Flux_Info_a0998[i][1]) + sum(Flux_Info_a0998[i][2]))
    
    Eps_998 = [(5/32) * 1 * ((r_998[i])**(3/2) + (0.998))**(10/3)*Flux_0998_GW[i] for i in range(0,len(Flux_Info_a0998))]

    r_97,Flux_097_GW = [],[]
    for i in range(0,len(Flux_Info_a097)):
        r_97.append(Flux_Info_a097[i][0])
        Flux_097_GW.append(sum(Flux_Info_a097[i][1]) + sum(Flux_Info_a097[i][2])) #typo in niels data.
    Eps_97 = [(5/32) * 1 * ((r_97[i])**(3/2) + (0.98))**(10/3)*Flux_097_GW[i] for i in range(0,len(Flux_Info_a097))]    
   
    r_a4,Flux_a4_GW = [],[]
    for i in range(0,len(Flux_Info_a4)):
        r_a4.append(Flux_Info_a4[i][0])  # Radial coordinates
        Flux_a4_GW.append(sum(Flux_Info_a4[i][1]) + sum(Flux_Info_a4[i][2]))  # Total energy flux
    
    Eps_a4 = [(5/32) * 1 * ((r_a4[i])**(3/2) + (1-10**-4))**(10/3)*Flux_a4_GW[i] for i in range(0,len(Flux_Info_a4))]
    
    r_a09995,Flux_a09995_GW = [],[]
    for i in range(0,len(Flux_Info_a09995)):
        r_a09995.append(Flux_Info_a09995[i][0])  # Radial coordinates
        Flux_a09995_GW.append(sum(Flux_Info_a09995[i][1]) + sum(Flux_Info_a09995[i][2]))  # Total energy flux
    
    Eps_a09995 = [(5/32) * 1 * ((r_a09995[i])**(3/2) + (0.9995))**(10/3)*Flux_a09995_GW[i] for i in range(0,len(Flux_Info_a09995))]
    
    r_a5,Flux_a5_GW = [],[]
    for i in range(0,len(Flux_Info_a5)):
        r_a5.append(Flux_Info_a5[i][0])  # Radial coordinates
        Flux_a5_GW.append(sum(Flux_Info_a5[i][1]) + sum(Flux_Info_a5[i][2]))  # Total energy flux
    
    Eps_a5 = [(5/32) * 1 * ((r_a5[i])**(3/2) + (1-10**-5))**(10/3)*Flux_a5_GW[i] for i in range(0,len(Flux_Info_a5))]
    
   
    
    
    return Eps_97,r_97,Eps_998,r_998,Eps_999999999,r_999999999,r_a4,Eps_a4,r_a09995 ,Eps_a09995, r_a5,Eps_a5 
def Extrapolate(a):

    from scipy import interpolate
    """
    This function will take the data from the function above and create an 
    interpolating function for the general relativistic corrections. These are
    the general relativistic corrections to the total energy flux due to gravitational wave emission.
    This function is used for the trajectories and, ultimately, the frequency evolution
    of the signal. This function is extremely key....    
    """
    
    
    """
    Jonathan, if there are any total screw ups in the code or waveform model I think they will
    spawn from the mess below. You can decide though.
    
    I need to improve the waveform model. I should build a universal interpolant that is valid 
    for spins a \in 1 - 10**(-2,-13).
    """                
    Eps_97,r_97,Eps_998,r_998,Eps_ext,r_ext,r_a4,Eps_a4,r_a09995 ,Eps_a09995, r_a5,Eps_a5 = EpsRelationship() # Unpack data values
    
    r_ext[0] = risco(1-10**-9)
    if a > 1-10**-9:
        r_isco_ext = risco(a)
        r_horiz = 1 + np.sqrt(1-a**2)
        EpsExt = (427/3200) *(r_isco_ext**(3/2) + a)**(10/3) * (r_isco_ext-r_horiz)/r_horiz    
        Eps_ext = sorted([EpsExt] + Eps_ext)
        r_ext = sorted([r_isco_ext] + r_ext)
        
    EpsFun = interpolate.interp1d(r_ext,Eps_ext,kind = 'cubic') 
    

    return EpsFun    


        

    
def ExtrapolateInf_All(a):
    """
    This function is going to try and create the most perfect 1-10**-9 waveform. Hopefully, it will calculate all of the 
    different relativistic correction values for each mode from m = 2 to 15. I could, perhaps, only extract
    the relevant modes up to m = 10. So that I have m \in {1,2,...,9,10}. Could be rather tasty, actually.
    
    If I do only take the modes up to m = 10, does this give a good approximation to the total flux? 
    I.e., could I show that what I am left with is negligable? Perhaps this could be good enough reason
    to ignore the other flux terms!
    
    Edit: Done, overlap with extra harmonic is 0.999995.
    
    These harmonics directly influence the amplitude of the signal. They are not
    
    
    """
    
    Flux_999999999_Inf, r_999999999, Flux_0998_Inf, r_998, Flux_097_Inf, r_97 =  Flux_Data()
    # Read in the data above
    Flux_098_Inf,r_98 = Flux_097_Inf, r_97
    Eps_Inf_ext = []
    Eps_Inf_998 = []
    Eps_Inf_098 = []
    
    # =============================================================================
    # #    What this gross piece of code does is calculates the infinity flux at a SPECIFIC
    #  Radial coordinate (say r_isco) and returns the infinity fluxes at each of the harmonics
    #  m = 2,3,4, ...
    # The index r is the radial coordinate and the index m is the mode. 
    # =============================================================================
    
    
    for r in range(0,len(r_999999999)):   # Loop through all data points for r_999999999
        Eps_Inf_ext.append([np.array(Flux_999999999_Inf[r][m])/(Factor_E(m+2)*\
                                 Omega(r_999999999[r],1-10**-9,m+2)) for m in range(0,14)])
    for r in range(0,len(r_998)):
        Eps_Inf_998.append([np.array(Flux_0998_Inf[r][m])/(Factor_E(m+2)*\
                                 Omega(r_998[r],0.998,m+2)) for m in range(0,12)])
        
    for r in range(0,len(r_97)):
        Eps_Inf_098.append([np.array(Flux_098_Inf[r][m])/(Factor_E(m+2)*\
                                 Omega(r_98[r],0.98,m+2)) for m in range(0,12)])
    
    
    
    
    Interpolating_Eps_Inf_Ext_Functions = []
    Interpolating_Eps_Inf_998_Functions = [] 
    Interpolating_Eps_Inf_98_Functions = []
    Interpolating_Eps_Inf_Low_Functions = []
    
     # The above are empty lists that will eventually be appended to by different
     # interpolants corresponding to interpolations of the general relativistic corrections
     # for each harmonic
    
    Eps_r_coord = []  # Here we will add on the general relativistic corrections
    Eps_r_coord_998 = []
    Eps_r_coord_098 = []
    
    r_horiz = 1 + np.sqrt(1-a**2)
    for m in range(0,12):  
        Eps_r_coord_998.append([Eps_Inf_998[r][m] for r in range(0,len(r_998))])
        Eps_r_coord.append([Eps_Inf_ext[r][m] for r in range(0,len(r_999999999))]) # For each harmonic, append general relativistic corrections
        Eps_r_coord_098.append([Eps_Inf_098[r][m] for r in range(0,len(r_98))])

    rhoriz_ext = 1 + np.sqrt(1-a**2)
    r_999999999[0] = risco(1-10**-9)
    
    for m in range(0,12):
        Interpolating_Eps_Inf_Ext_Functions.append(interpolate.InterpolatedUnivariateSpline([1] + r_999999999,[0] + Eps_r_coord[m],k = 3))
            # We now append our interpolants for each harmonic to the list above.
#    elif a >= 0.99 and a < 1-10**-9:            
#        for m in range(0,12):  
#            Interpolating_Eps_Inf_998_Functions.append(interpolate.InterpolatedUnivariateSpline([r_horiz] + r_998,[0] + Eps_r_coord_998[m],k = 3))
#    elif a > 0.9 and a < 0.99:
#        for m in range(0,12):  
#            Interpolating_Eps_Inf_98_Functions.append(interpolate.InterpolatedUnivariateSpline([r_horiz] + r_98,[0] + Eps_r_coord_098[m],k = 3))
   
    Interpolating_Eps_Inf_998_Functions = 0
        
    Interpolating_Eps_Inf_Functions = Interpolating_Eps_Inf_Ext_Functions

    return Interpolating_Eps_Inf_Functions

def Check_Interpolant():
    a = 0.9999
    EpsFun_approximate = Extrapolate(a)
    Eps_97,r_97,Eps_998,r_998,Eps_999999999,r_999999999,r_a4,Eps_a4,r_a09995 ,Eps_a09995, r_a5,Eps_a5 = EpsRelationship()  
    
    EpsFun_exact = interpolate.interp1d(r_998,Eps_998,kind = 'cubic')
    
    r = np.arange(r_998[0],2,0.0001)
    EpsFun_approx_val = EpsFun_approximate(r)
    
    EpsFun_exact_val = EpsFun_exact(r)
    
    plt.plot(r,EpsFun_approx_val,'r--',label = 'approx interpolant')
    plt.plot(r,EpsFun_exact_val,'b',label = 'exact interpolant for a = 0.998')
    plt.xlabel(r'$\tilde{r}$')
    plt.ylabel(r'$\dot{\mathcal{E}}$')
    plt.legend()
    plt.show()
    plt.clf()
    
    error = EpsFun_exact_val/EpsFun_approx_val
    
    plt.plot(r,error)
    plt.show()
    plt.clf()
    
    # =============================================================================
    #     Now check with a larger spin of a = 0.9999
    # =============================================================================
    
    r_a4[0] = risco(0.9999)
    EpsFun_a4 = interpolate.interp1d(r_a4,Eps_a4,kind = 'cubic')
    
    
    r = np.arange(r_a4[0],2,0.0001)
    EpsFun_a4_val = EpsFun_a4(r)
    
    plt.plot(r,EpsFun_a4_val,'k--',label = 'exact a = 0.9999')
    plt.plot(r,EpsFun_approximate(r),'darkviolet',label = 'approximate')
    plt.xlabel(r'$\tilde{r}$')
    plt.ylabel(r'$\dot{\mathcal{E}}$')
    plt.legend()
    plt.show()
    plt.clf()
    
    error = EpsFun_a4_val / EpsFun_approximate(r)
    
    plt.plot(r,error)
    plt.title('Error in a = 0.9999')
    plt.xlabel(r'$\tilde{r}$')
    plt.ylabel(r'Fractional Error')
    plt.show()
    plt.clf()
    
    # =============================================================================
    #     Now check with a spin of a = 0.9995 - pray
    # =============================================================================
    
    r_a09995[0] = risco(0.9995)
    EpsFun_a09995 = interpolate.interp1d(r_a09995,Eps_a09995,kind = 'cubic')
    
    
    r = np.arange(r_a09995[0],2,0.0001)
    EpsFun_a09995_val = EpsFun_a09995(r)
    
    plt.plot(r,EpsFun_a09995_val,'k--',label = 'exact a = 0.9995')
    plt.plot(r,EpsFun_approximate(r),'darkviolet',label = 'approximate')
    plt.xlabel(r'$\tilde{r}$')
    plt.ylabel(r'$\dot{\mathcal{E}}$')
    plt.legend()
    plt.show()
    plt.clf()
    
    error = EpsFun_a09995_val / EpsFun_approximate(r)
    
    plt.plot(r,error)
    plt.title('Error in a = 0.9999')
    plt.xlabel(r'$\tilde{r}$')
    plt.ylabel(r'Fractional Error')
    plt.show()
    plt.clf()
    # =============================================================================
    #     Now check with a spin of a = 0.99
    # =============================================================================
    Eps99 = [0.4148,0.4154,0.4160,0.4177,0.4207,0.4263,0.4434,0.4701,0.5182,
             0.5587,0.5930,0.6665,0.7117,0.7556,0.7813,0.8121,0.8320,0.8469,
             0.8589,0.8689,0.8774,0.8847]
        
    R =[1.000,1.001,1.002,1.005,1.01,1.02,1.05,1.1,1.2,1.3,
         1.4,1.7,2.0,2.5,3,4,5,6,7,8,9,10] 
    
    r_isco99 = risco(0.99)
    
    r_99 = r_isco99 * np.array(R)
    
    EpsFun_99 = interpolate.interp1d(r_99,Eps99,kind = 'cubic')  
    r = np.arange(r_isco99,2,0.0001)
    
    plt.plot(r,EpsFun_99(r),'k--',label = r'$a = 0.99$')
    plt.plot(r,EpsFun_approximate(r),'darkviolet',label = r'approximate')
    plt.xlabel(r'$\tilde{r}$')
    plt.ylabel(r'$\dot{\mathcal{E}}$')
    plt.legend()
    plt.show()
    plt.clf()
    
    error = EpsFun_99(r) / EpsFun_approximate(r)
    
    plt.plot(r,error)
    plt.title('Error in a = 0.999')
    plt.xlabel(r'$\tilde{r}$')
    plt.ylabel(r'Fractional Error')
    plt.show()
    plt.clf()
    # =============================================================================
    #   and with a spin of a = 0.999
    # =============================================================================
    Eps999 = [0.2022,0.2032,0.2041,0.2069,0.2116,0.2208,0.2473,0.2881,0.3581,
       0.4160,0.4648,0.5723,0.6411,0.7089,0.7469,0.7882,0.8118,0.8286,
       0.8416,0.8524,0.8616,0.8695]
        
        
    R =[1.000,1.001,1.002,1.005,1.01,1.02,1.05,1.1,1.2,1.3,
         1.4,1.7,2.0,2.5,3,4,5,6,7,8,9,10] 
    
    r_isco999 = risco(0.999)
    
    r_999 = r_isco999 * np.array(R)
    
    EpsFun_999 = interpolate.interp1d(r_999,Eps999,kind = 'cubic')  
    r = np.arange(r_isco999,2,0.0001)
    
    plt.plot(r,EpsFun_999(r),'k--',label = r'$a = 0.999$')
    plt.plot(r,EpsFun_approximate(r),'darkviolet',label = r'approximate')
    plt.xlabel(r'$\tilde{r}$')
    plt.ylabel(r'$\dot{\mathcal{E}}$')
    plt.legend()
    plt.show()
    plt.clf()
    
    error = EpsFun_999(r) / EpsFun_approximate(r)
    
    plt.plot(r,error)
    plt.title('Error in a = 0.999')
    plt.xlabel(r'$\tilde{r}$')
    plt.ylabel(r'Fractional Error')
    plt.show()
    plt.clf()
    
    # =============================================================================
    #   what about a spin of a = 0.99999
    # =============================================================================
    
    r_a5[0] = risco(0.99999)
    Eps_a5.pop(2);Eps_a5.pop(5)
    r_a5.pop(2);r_a5.pop(5)
    
    
    EpsFun_a5 = interpolate.interp1d(r_a5,Eps_a5,kind = 'cubic')
    
    
    r = np.arange(r_a5[0],2,0.0001)
    EpsFun_a5_val = EpsFun_a5(r)
    
    plt.plot(r,EpsFun_a5_val,'k--',label = 'exact a = 0.99999')
    plt.plot(r,EpsFun_approximate(r),'darkviolet',label = 'approximate')
    plt.xlabel(r'$\tilde{r}$')
    plt.ylabel(r'$\dot{\mathcal{E}}$')
    plt.legend()
    plt.show()
    plt.clf()
    
    error = EpsFun_a5_val / EpsFun_approximate(r)
    
    plt.plot(r,error)
    plt.title('Error in a = 0.99999')
    plt.xlabel(r'$\tilde{r}$')
    plt.ylabel(r'Fractional Error')
    plt.show()
    plt.clf()

def zoom():
    import matplotlib.pyplot as plt
    import numpy as np
    a = 1-10**-9
    EpsFun_approximate = Extrapolate(a)
    Eps_97,r_97,Eps_998,r_998,Eps_999999999,r_999999999,r_a4,Eps_a4,r_a09995 ,Eps_a09995, r_a5,Eps_a5 = EpsRelationship()  
    
    EpsFun_exact = interpolate.interp1d(r_998,Eps_998,kind = 'cubic')
    r_ext = np.arange(r_999999999[0],2,0.0001)
    
    r0998 = np.arange(r_998[0],2,0.0001)
    EpsFun_interp = EpsFun_approximate(r_ext)
    
    EpsFun_exact_val = EpsFun_exact(r0998)
    

    r_a5[0] = risco(0.99999)
    Eps_a5.pop(2);Eps_a5.pop(5)
    r_a5.pop(2);r_a5.pop(5)
    
    
    EpsFun_a5 = interpolate.interp1d(r_a5,Eps_a5,kind = 'cubic')
    
    
    ra5 = np.arange(r_a5[0],2,0.0001)
    EpsFun_a5_val = EpsFun_a5(ra5)
    
    



    
    # =============================================================================
    #     Now check with a larger spin of a = 0.9999
    # =============================================================================
    
    r_a4[0] = risco(0.9999)
    EpsFun_a4 = interpolate.interp1d(r_a4,Eps_a4,kind = 'cubic')
    
    
    ra4 = np.arange(r_a4[0],2,0.0001)
    EpsFun_a4_val = EpsFun_a4(ra4)

    
    # =============================================================================
    #     Now check with a spin of a = 0.9995 - pray
    # =============================================================================
    
    r_a09995[0] = risco(0.9995)
    EpsFun_a09995 = interpolate.interp1d(r_a09995,Eps_a09995,kind = 'cubic')
    
    
    ra09995 = np.arange(r_a09995[0],2,0.0001)
    EpsFun_a09995_val = EpsFun_a09995(ra09995)
    






    # =============================================================================
    #     Now check with a spin of a = 0.99
    # =============================================================================

    # =============================================================================
    #   and with a spin of a = 0.999
    # =============================================================================
    Eps999 = [0.2022,0.2032,0.2041,0.2069,0.2116,0.2208,0.2473,0.2881,0.3581,
       0.4160,0.4648,0.5723,0.6411,0.7089,0.7469,0.7882,0.8118,0.8286,
       0.8416,0.8524,0.8616,0.8695]
        
        
    R =[1.000,1.001,1.002,1.005,1.01,1.02,1.05,1.1,1.2,1.3,
         1.4,1.7,2.0,2.5,3,4,5,6,7,8,9,10] 
    
    r_isco999 = risco(0.999)
    
    r_999 = r_isco999 * np.array(R)
    
    EpsFun_999 = interpolate.interp1d(r_999,Eps999,kind = 'cubic')  
    r999 = np.arange(r_isco999,2,0.0001)




    
    # =============================================================================
    #   what about a spin of a = 0.99999
    # =============================================================================
    
                                                  
    Eps99 = [0.4148,0.4154,0.4160,0.4177,0.4207,0.4263,0.4434,0.4701,0.5182,
             0.5587,0.5930,0.6665,0.7117,0.7556,0.7813,0.8121,0.8320,0.8469,
             0.8589,0.8689,0.8774,0.8847]
        
    R =[1.000,1.001,1.002,1.005,1.01,1.02,1.05,1.1,1.2,1.3,
         1.4,1.7,2.0,2.5,3,4,5,6,7,8,9,10] 
    
    r_isco99 = risco(0.99)
    
    r_99 = r_isco99 * np.array(R)
    
    EpsFun_99 = interpolate.interp1d(r_99,Eps99,kind = 'cubic')  
    r99 = np.arange(r_isco99,2,0.0001)
    fig, ax = plt.subplots() # create a new figure with a default 111 subplot   
    



 
#    ax.plot(r99,EpsFun_99(r99),'y',label = r'$a = 0.99$')
#    ax.plot(r0998,EpsFun_exact_val,'b',label = r'$a = 0.998$')     
    ax.plot(r999,EpsFun_999(r999),'c',label = r'$a = 0.999$')    
    ax.plot(ra09995,EpsFun_a09995_val,'k',label = r'$a = 0.9995$')    
    ax.plot(ra4,EpsFun_a4_val,'g',label = r'$a = 0.9999$')
    ax.plot(ra5,EpsFun_a5_val,'m',label = r'$a = 0.99999$')    
    ax.plot(r_ext,EpsFun_interp,'k--',label = 'interpolant')  
    ax.legend(loc = 'lower right',prop = {'size':19})
    
    ax.set_xlabel(r'$\tilde{r}$')
    ax.set_ylabel(r'$\dot{\mathcal{E}}$') 
    ax.set_title('High Spin General Relativistic Corrections')


