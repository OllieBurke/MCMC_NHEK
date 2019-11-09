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


"""
These are the changes to my flux data.
"""

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
    return r_isco,a   # In later functions, r_isco, a and x_isco is used.

def Omega(r,a,m):
    # Particles dimensionless angular frequency
    return ((r**(3/2) + a)**-1)**(2 + 2*m/3)

def Paths():
    """
    Underneath are the hardcoded paths. You simply need to change these to the
    working directories of where the files a0.99999999, a0.998,a0.97. 
    """
    path_Ext = '/Users/Ollie/Documents/Thesis/NHEK_Geometry/NHEK_Code/Gargantua_data/a0.999999999'
    path_a0998 = '/Users/Ollie/Documents/Thesis/NHEK_Geometry/NHEK_Code/Gargantua_data/a0.998'
    path_a097 = '/Users/Ollie/Documents/Thesis/NHEK_Geometry/NHEK_Code/Gargantua_data/a0.97'
    Home_Of_Code = '/Users/Ollie/Documents/Thesis/NHEK_Geometry/NHEK_Code'
    
    # Jonathan, put the location of each of your directories here:

#    path_Ext = '/home/mattedwards/Ollie/Gargantua_data/a0.999999999'
#    path_a0998 = '/home/mattedwards/Ollie/Gargantua_data/a0.998'
#    path_a097 = '/home/mattedwards/Ollie/Gargantua_data/a0.97'
#    Home_Of_Code = '/home/mattedwards/Ollie/'

    
    return path_Ext,path_a0998,path_a097,Home_Of_Code 

def Flux_Data():
    """
    This function reads in each of the flux data that was given to me by Niels
    back in the day. Hello there. 
    """
    path_Ext,path_a0998,path_a097,Home_Of_Code = Paths()

    os.chdir(path_Ext)
    import ReadFluxes_a0999999999
    Flux_Info_a0999999999 = ReadAllData_a0999999999() # Read in near-extremal flux data
        
    os.chdir(path_a0998)
    import FluxData_a0998
    Flux_Info_a0998 = ReadAllData_a0998()  # Read in 0.998 flux data
    
    os.chdir(path_a097) # Changes working directory
    import FluxData_a097  # Import the script file
    Flux_Info_a097 = ReadAllData_a097()   # Read in all of the data flux data. This is the same for above.
                                          # NOTE: Typo in Niels Data. a097 is actually a098.
        
    os.chdir(Home_Of_Code)
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
    # ----- Import all of Thornes data values
    Eps999 = [0.2022,0.2032,0.2041,0.2069,0.2116,0.2208,0.2473,0.2881,0.3581,
           0.4160,0.4648,0.5723,0.6411,0.7089,0.7469,0.7882,0.8118,0.8286,
           0.8416,0.8524,0.8616,0.8695]
           
    Eps99 = [0.4148,0.4154,0.4160,0.4177,0.4207,0.4263,0.4434,0.4701,0.5182,
             0.5587,0.5930,0.6665,0.7117,0.7556,0.7813,0.8121,0.8320,0.8469,
             0.8589,0.8689,0.8774,0.8847]
             
    Eps9 = [0.7895,0.7894,0.7894,0.7891,0.7887,0.7880,0.7867,0.7859,0.7882,
            0.7920,0.7960,0.8075,0.8171,0.8302,0.8415,0.8597,0.8742,0.8858,
            0.8955,0.9036,0.9105,0.9164]
            
    Eps8 = [0.9144,0.9140,0.9137,0.9126,0.9109,0.9076,0.8988,0.8876,0.8726,
            0.8638,0.8583,0.8524,0.8530,0.8589,0.8662,0.8807,0.8930,
            0.9031,0.9116,0.9186,0.9246,0.9298]
            
    Eps5 = [1.053,1.053,1.052,1.050,1.047,1.041,1.025,1.002,0.9706,0.9493,
            0.9348,0.9119,0.9034,0.9012,0.9040,0.9129,0.9216,0.9291,
            0.9354,0.9407,0.9452,0.9491]
            
    Eps2 = [1.114,1.114,1.113,1.111,1.107,1.100,1.081,1.055,1.017,0.9913,
            0.9733,0.9435,0.9312,0.9248,0.9250,0.9306,0.9371,0.9430,0.9480,
            0.9522,0.9558,0.9589]
            
    Eps0 = [1.143,1.142,1.141,1.139,1.135,1.127,1.108,1.080,1.039,1.012,
            0.9919,0.9591,0.9448,0.9363,0.9352,0.9391,0.9448,0.9490,
            0.9533,0.9588,0.9607,0.9616]

    EpsNeg5 = [1.197,1.196,1.196,1.193,1.189,1.181,1.159,1.128,1.082,1.051,1.028,
               0.9888,0.9705,0.9580,0.9542,0.9546,0.9577,0.9611,0.9641,0.9669,0.9693,
               0.9714]

    EpsNeg9 = [1.233,1.232,1.231,1.228,1.224,1.215,1.192,1.159,1.110,1.075,1.051,1.007,
               0.9862,0.9709,0.9653,0.9634,0.9651,0.9675,0.9699,0.9720,0.9740,0.9757]
    
    EpsNeg99 = [1.240,1.239,1.238,1.235,1.231,1.222,1.198,1.165,1.115,1.081,1.055,
                1.011,0.9893,0.9734,0.9674,0.9651,0.9665,0.9687,0.9709,0.9730,0.9749,
                0.9765]

    R =[1.000,1.001,1.002,1.005,1.01,1.02,1.05,1.1,1.2,1.3,
         1.4,1.7,2.0,2.5,3,4,5,6,7,8,9,10]   # Import Thornes R value.

             
                     # These values are added after the R values are computed.
                     # They add in extra data values once we move into the a = ... regime.
    Eps99List = Eps99[-2:]   
    Eps9List = Eps9[-4:]
    Eps8List = Eps8[-2:]
    Eps5List = Eps5[-3:]
    Eps2List = Eps2[-2:]
    Eps0List = Eps0[-1:]
    EpsNeg5List = EpsNeg5[-2:]
    EpsNeg9List = EpsNeg9[-2:]
    EpsNeg99List = EpsNeg99[-1:]
    
    r_isco = risco(0.999,1,1)[0]
    RxR_isco999 = [r*r_isco for r in R]

    r_isco = risco(0.99,1,1)[0]
    RxR_isco99 = [r*r_isco for r in R]
    R99new = RxR_isco99[-2:] 
    
    r_isco = risco(0.9,1,1)[0]
    RxR_isco9 = [r*r_isco for r in R]
    R9new = RxR_isco9[-4:]

    r_isco = risco(0.8,1,1)[0]
    RxR_isco8 = [r*r_isco for r in R]
    R8new = RxR_isco8[-2:]
    
    r_isco = risco(0.5,1,1)[0]
    RxR_isco5 = [r*r_isco for r in R]
    R5new = RxR_isco5[-3:]

    r_isco = risco(0.2,1,1)[0]
    RxR_isco2 = [r*r_isco for r in R]
    R2new = RxR_isco2[-2:]

    
    r_isco = risco(0,1,1)[0]
    RxR_isco0 = [r*r_isco for r in R]
    R0new = RxR_isco0[-1:]
    
    r_isco = risco(-0.5,1,1)[0]
    RxR_iscoNeg5 = [r*r_isco for r in R]
    RNeg5New = RxR_iscoNeg5[-2:]

    r_isco = risco(-0.9,1,1)[0]
    RxR_iscoNeg9 = [r*r_isco for r in R]
    RNeg9New = RxR_iscoNeg9[-2:]

    r_isco = risco(-0.99,1,1)[0]
    RxR_iscoNeg99 = [r*r_isco for r in R]
    RNeg99New = RxR_iscoNeg99[-1:]
       
 
    path_Ext,path_a0998,path_a097,Home_Of_Code = Paths()                   
    os.chdir(path_Ext)
    import ReadFluxes_a0999999999
    Flux_Info_a0999999999 = ReadAllData_a0999999999()
        
    os.chdir(path_a0998)
    import FluxData_a0998
    Flux_Info_a0998 = ReadAllData_a0998()
    
    os.chdir(path_a097) # Changes working directory
    import FluxData_a097  # Import the script file
    Flux_Info_a097 = ReadAllData_a097()   # Read in all of the data flux data. This is the same for above.
        
    os.chdir(Home_Of_Code)
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
    
    Eps_998 = [(5/32) * 1 * ((r_998[i])**(3/2) + (1-10**-9))**(10/3)*Flux_0998_GW[i] for i in range(0,len(Flux_Info_a0998))]

    r_97,Flux_097_GW = [],[]
    for i in range(0,len(Flux_Info_a097)):
        r_97.append(Flux_Info_a097[i][0])
        Flux_097_GW.append(sum(Flux_Info_a097[i][1]) + sum(Flux_Info_a097[i][2]))
    
    Eps_97 = [(5/32) * 1 * ((r_97[i])**(3/2) + (1-10**-9))**(10/3)*Flux_097_GW[i] for i in range(0,len(Flux_Info_a097))]


    
    R999999999 = r_999999999   # Relabeling.
    

    CombineEpsThorne_999999999 = (list((Eps_999999999))) # Import relativistic correction values for near-extremal spins.
    
    Rlist = RxR_isco999  + R99new  + R9new + R8new + R5new + R2new + R0new \
                    + RNeg5New + RNeg9New + RNeg99New   # Form list of R coordinates.
#    
#
    CombineEpsThorne = (list((Eps999  + Eps99List + Eps9List + Eps8List + \
                              Eps5List + Eps2List + Eps0List + EpsNeg5List + EpsNeg9List  + EpsNeg99List))) 
    
    # Combine all of Thorne's data.
    

   
    
    
    return Rlist,CombineEpsThorne,Eps_97,r_97,Eps_998,r_998,Eps_999999999,r_999999999 

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
    """                
    Rlist,CombineEpsThorne,Eps_97,r_97,Eps_998,r_998,Eps_999999999,r_999999999 = EpsRelationship() # Unpack data values
        
                                                                                            # a.
#    if (a>0.999999999 and a <= 1):
#        print(a)
#        r_horiz = 1 + (1-a**2)**(1/2)
#        R_near = r_horiz + 1e-15
#        EpsExt = (427/3200) *(R_near**(3/2) + a)**(10/3) * (R_near-r_horiz)/r_horiz 
#        
#        
#        EpsFinal =  [EpsExt] + Eps_999999999
#        RFinal =  [R_near] + r_999999999
#        RFinal.sort()
#        EpsFinal.sort()
#        EpsFun = interpolate.interp1d([r_horiz] +  RFinal, [0]  + EpsFinal,kind = 'cubic') 
              
    if a > 1-10**-9:
    
        """
        For near-extremal spins a > 1-10**-9
        """
        print('You are using the near-extremal interpolant - a>1-10^{-9}')
        r_isco = risco(a,1,1)[0]   # Find ISCO
        r_horiz = 1+np.sqrt(1-a**2)  # Calculate Horizon
        EpsExt = (427/3200) *(r_isco**(3/2) + a)**(10/3) * (r_isco-r_horiz)/r_horiz
        
        # Here we use the near-extremal approximation to the flux. We use the
        # Gralla et al flux formula for near-extremal black holes. We 
        # initialise the gradient by using a single point (at the ISCO, where
        # the formula is still defined.
        
        R_New = [r_horiz] + [r_isco] + r_999999999
        Eps_New = [0] + [EpsExt] + Eps_999999999
        
        # We know that the total energy flux at the horizon is zero. 
        # As a result, we can add an extra point at r,eps(r) -> (rhoriz,0).
        
        EpsFun = interpolate.interp1d(R_New,Eps_New,kind = 'cubic') 
        
        # Interpolate
        
    
            
    elif a == 0.999999999:
        print('You are using the (given) near-extremal interpolant - a = 1-10^{-9}')
        # For identically 1-10^{-9}. This should be exact for a circular, equatorial
        # inspiral with the spin defined by above.
        
        r_horiz = 1 + np.sqrt(1-a**2)  # Horizon
        r_isco = risco(1-10**-9,1,1)[0] # ISCO
        r_999999999[0] = r_isco #redefine the first value in r_99... to be the 
                                # exact ISCO value 
        
        R_New = [r_horiz] + r_999999999  
        Eps_New =  [0] + Eps_999999999
        
        EpsFun = interpolate.interp1d(R_New, Eps_New,kind = 'cubic')
    
    
        
       
    
    elif (a>0.999 and a<0.999999999): 
        """
        I think that this is when things are going to start getting rocky.
        I am basically using the flux data given by thorne for a spin of 
        a = 0.999 and then using the near-extremal approximation to find an 
        extra point close to the ISCO. I'm not sure how good an approximation this
        is though... is a = 0.999 near-extremal enough? Stay tuned.
        
        """
        print('You are using the interpolant for between 0.999 and 1-10^-9')
        
        Eps999 = [0.2022,0.2032,0.2041,0.2069,0.2116,0.2208,0.2473,0.2881,0.3581,
           0.4160,0.4648,0.5723,0.6411,0.7089,0.7469,0.7882,0.8118,0.8286,
           0.8416,0.8524,0.8616,0.8695]  # Thorne data
        
        r_isco999 = risco(0.999,1,1)[0]  # ISCO
        r_isco_ext = risco(a,1,1)[0]  # extremal ext ISCO < r_isco999
        R = r_isco999 * np.array([1.000,1.001,1.002,1.005,1.01,1.02,1.05,1.1,1.2,1.3,
             1.4,1.7,2.0,2.5,3,4,5,6,7,8,9,10])
        r_horiz = 1 + np.sqrt(1-a**2)  # Horizon
        R_near = r_isco_ext + 1e-10  # Horizon + small perturbation (so we don't get zero in formula below)
        EpsExt_999 = (427/3200) *(R_near**(3/2) + 0.999)**(10/3) * (R_near-r_horiz)/r_horiz
        #Calculate the general relativistic correction values when extremely close to the horizon
        # again, what we are doing here is giving the interpolant that extra point close to the 
        # horizon so that the formula is as "valid" as possible. 
        
        Rfinal = [r_horiz] + [R_near] + list(R)  # self explanatory
        EpsFinal = [0] + [EpsExt_999] + Eps999  # self explanatory
        EpsFun = interpolate.interp1d(Rfinal,EpsFinal,kind = 'cubic') # Cubic interpolant.
        
    elif a == 0.999:
        print('You are now using Thornes data for a = 0.999.')
        Eps999 = [0.2022,0.2032,0.2041,0.2069,0.2116,0.2208,0.2473,0.2881,0.3581,
           0.4160,0.4648,0.5723,0.6411,0.7089,0.7469,0.7882,0.8118,0.8286,
           0.8416,0.8524,0.8616,0.8695] # Thorne's data
        
        r_isco999 = risco(0.999,1,1)[0]
        R = r_isco999 * np.array([1.000,1.001,1.002,1.005,1.01,1.02,1.05,1.1,1.2,1.3,
             1.4,1.7,2.0,2.5,3,4,5,6,7,8,9,10])  # Radial coordinates
        r_horiz_999 = 1 + np.sqrt(1-0.999**2) # Horizon

        EpsFun = interpolate.interp1d([r_horiz_999]  + list(R), [0]  + Eps999,kind = 'cubic') 
        # The above forms an interpolant that is fit for spins a = 0.999
    
    elif a == 0.998:
#        print('You are using the interpolant for a spin of a = 0.998')
        r_isco_998 = risco(0.998,1,1)[0]  # Calculate ISCO
        r_998[0] = r_isco_998 # Replace first value of r_998 with ISCO
        EpsFun = interpolate.interp1d([1 + np.sqrt(1-a**2)]+r_998,[0] + Eps_998,kind = 'cubic') # Calculate interpolant
    elif a == 0.99:
        print('Use exact flux data a = 0.99')
        Eps99 = [0.7895,0.7894,0.7894,0.7891,0.7887,0.7880,0.7867,0.7859,0.7882,
                0.7920,0.7960,0.8075,0.8171,0.8302,0.8415,0.8597,0.8742,0.8858,
                0.8955,0.9036,0.9105,0.9164] # Exact thorne flux data
        
        r_isco9 = risco(0.9,1,1)[0]
        R = r_isco9 * np.array([1.000,1.001,1.002,1.005,1.01,1.02,1.05,1.1,1.2,1.3,
             1.4,1.7,2.0,2.5,3,4,5,6,7,8,9,10])  
        r_horiz_99 = 1 + np.sqrt(1-0.99**2)

    
        EpsFun = interpolate.interp1d([r_horiz_99] + list(R), [0]  + Eps99,kind = 'cubic')
    
    
    elif a == 0.98:
        print('Use exact data given by Niels for a = 0.98')
        ''' 
        typo in Niels data, he gave me spins for 0.98 <--- remember
        '''
        
        r_low_spin = r_97 + r_998[15:]  # Here we use the radial coordinates
                                        # given by r_998 to extend the range
                                        # of the interpolant
        Eps_low_spin = Eps_97 + Eps_998[15:] # Similar story.
    
     
        EpsFun = interpolate.interp1d([1 + np.sqrt(1-0.98**2)] + r_low_spin,[0] + Eps_low_spin,'cubic') #~
        EpsVal = EpsFun(r)
    
    
    elif a <= 0.99 and a > 0.9:
        print('Now you are using the interpolant for a spin between 0.9 and 0.99 ')
        r_isco = risco(a,1,1)[0]  # Calculate r_isco99
        NewEps = [item for item in CombineEpsThorne if item >=0.4148]
        del Rlist[0:9] # Delete the first 9 values in Rlist
#        NewEps += [0.4148]
#        Rlist += [r_isco]
#        del Rlist[0]
#        del NewEps[0]
        NewEps.sort()
        Rlist.sort()
    
        
        
    # =============================================================================
    #         True Eps9 From Thorne
    # =============================================================================
                
    # =============================================================================
    #         True Eps99 from Thorne
    # =============================================================================
        
        Eps99 = [0.4148,0.4154,0.4160,0.4177,0.4207,0.4263,0.4434,0.4701,0.5182,
             0.5587,0.5930,0.6665,0.7117,0.7556,0.7813,0.8121,0.8320,0.8469,
             0.8589,0.8689,0.8774,0.8847]  # Thorne's data
        
        
        R = r_isco * np.array([1.000,1.001,1.002,1.005,1.01,1.02,1.05,1.1,1.2,1.3,
         1.4,1.7,2.0,2.5,3,4,5,6,7,8,9,10])
        
        r_horiz_99 = 1 + np.sqrt(1-a**2) # Horizon
#        R_near = r_horiz_99 + 0.000000001  # Near the horizon to use with
            
#        EpsExt_99 = (427/3200) *(R_near**(3/2) + 0.999)**(10/3) * (R_near-r_horiz_99)/r_horiz_99 
    
        
        r_low_spin = r_97 + r_998[15:]  # Extend list of radial coordinates
        Eps_low_spin = Eps_97 + Eps_998[15:]  # Extend list of relativistic correction values
     
        EpsFun = interpolate.interp1d([1 + np.sqrt(1 - a**2)]  + r_low_spin,[0] + Eps_low_spin,kind = 'cubic') 
        # Create interpolant
        

      
    elif ((a>0.99) and (a<0.999)) and a!=0.998 and a!= 0.97 and a!= 0.99:
        print('if a between 0.99 and 0.999 and not equal to 0.998, 0.97 and 0.99 then use this interpolant')
        EpsFun = interpolate.interp1d([1 + np.sqrt(1-a**2)] + Rlist,[0] + CombineEpsThorne,kind = 'cubic') #~
    
    elif a==0.9:
        print('exact interpolant for a = 0.9')
        Eps9 = [0.7895,0.7894,0.7894,0.7891,0.7887,0.7880,0.7867,0.7859,0.7882,
                0.7920,0.7960,0.8075,0.8171,0.8302,0.8415,0.8597,0.8742,0.8858,
                0.8955,0.9036,0.9105,0.9164]
        
        r_isco9 = risco(0.9,1,1)[0]
        R = r_isco9 * np.array([1.000,1.001,1.002,1.005,1.01,1.02,1.05,1.1,1.2,1.3,
             1.4,1.7,2.0,2.5,3,4,5,6,7,8,9,10])
        r_horiz_9 = 1 + np.sqrt(1-0.9**2)
    
        EpsFun = interpolate.interp1d([r_horiz_9] + list(R), [0]  + Eps9,kind = 'cubic')
    
              
    elif (a>0.9) & (a<0.99):
        print('interpolant for anything between 0.9 and 0.99')
        EpsFun = interpolate.interp1d([1 + np.sqrt(1-a**2)] + Rlist,[0] + CombineEpsThorne,kind = 'cubic') #~

      
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
                                 Omega(r_999999999[r],1-10**-9,m+2)) for m in range(0,19)])
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


    if a>=1-1e-9:
        for m in range(0,12):
            Interpolating_Eps_Inf_Ext_Functions.append(interpolate.InterpolatedUnivariateSpline([r_horiz] + r_999999999,[0] + Eps_r_coord[m],k = 3))
            # We now append our interpolants for each harmonic to the list above.
    elif a >= 0.99 and a < 1-10**-9:            
        for m in range(0,12):  
            Interpolating_Eps_Inf_998_Functions.append(interpolate.InterpolatedUnivariateSpline([r_horiz] + r_998,[0] + Eps_r_coord_998[m],k = 3))
    elif a > 0.9 and a < 0.99:
        for m in range(0,12):  
            Interpolating_Eps_Inf_98_Functions.append(interpolate.InterpolatedUnivariateSpline([r_horiz] + r_98,[0] + Eps_r_coord_098[m],k = 3))
#    else:
#        r_low_spin = [r_horiz] + r_98 + r_998[15:]
#    
#        for m in range(0,12):  
#            Eps_low_spin = Eps_r_coord_098[m] + Eps_r_coord_998[m][15:]
#            Interpolating_Eps_Inf_Low_Functions.append(interpolate.InterpolatedUnivariateSpline(r_low_spin, [0] + Eps_low_spin,k = 3))      
  
    
    if a <= 0.98:
        """
        Here we use the (infinity) flux mode for spins a=0.98
        """
        print('You are using the infinity flux model (a=0.98) for low spins a < 0.98')

        Interpolating_Eps_Inf_Functions = Interpolating_Eps_Inf_98_Functions 
    elif a > 0.98 and a < 1-10**-9:
        """
        Here we use the flux model given to us by Niels for a = 0.998. 
        """
#        print('You are using the 0.998 infinity flux model')
        Interpolating_Eps_Inf_Functions = Interpolating_Eps_Inf_998_Functions
    elif a >= 1-10**-9:
        """
        We use the near-extremal flux data a = 1-10^-9 if we are interested
        in a spin a > 1-10^-6.
        """
        print("You are using the near-extremal flux given to us by Niels")
        Interpolating_Eps_Inf_Functions = Interpolating_Eps_Inf_Ext_Functions
        
    return Interpolating_Eps_Inf_Functions,Interpolating_Eps_Inf_998_Functions,Eps_Inf_ext,r_999999999

