#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file will attempt to read in the data given by Niels Warburton. This is for a spin of 
a = 1-10**-9
"""
import os
import re

def ReadAllData_a0999999999():
    """
    May need to be careful with this data. The derivative is smooth but rather strange. May need to 
    clean it when I have the time!
    """
    No_Files = len(os.listdir('/Users/Ollie/Documents/Thesis/NHEK_Geometry/NHEK_Code/Gargantua_data/a0.999999999'))
    Flux_Info_a0999999999 = []
    for i in range(0,No_Files):
        item = os.listdir('/Users/Ollie/Documents/Thesis/NHEK_Geometry/NHEK_Code/Gargantua_data/a0.999999999')[i]
        if item == ".DS_Store" or item == "ReadFluxes_a0999999999.py" or item == '__pycache__':
            continue
        file_name_flux = item
        r = float(re.findall("\d+\.\d+",item)[0])     # Radii the flux is calculated at 
    
        E_Inf_mode = [float(column.split(' ')[2]) for column in open(str(file_name_flux)).readlines()]
        E_Hor_mode = [float(column.split(' ')[3]) for column in open(str(file_name_flux)).readlines()]
                                        # ^ The above reads in the data as columns.
        
    
        E_Inf_Modes = []
        E_Hor_Modes = []
    
        for l in range(1,len(E_Inf_mode)):
            E_Inf_Modes.append(sum(E_Inf_mode[0:(l+1)]))  # Find the sum of the Inf modes up to m <= l
            E_Hor_Modes.append(sum(E_Hor_mode[0:(l+1)]))  # similar but for horizon.
            del E_Hor_mode[:(l+1)]
            del E_Inf_mode[:(l+1)]  # delete the modes once they've been entered into list.
            
        E_Inf_Modes = [item for item in E_Inf_Modes if item != 0]   # Remove the zeros (machine precision)
    
        E_Hor_Modes = [item for item in E_Hor_Modes if item != 0]
            
        
        Flux_Info_a0999999999.append([r,E_Inf_Modes,E_Hor_Modes])  # add the data in [_,_,_] for r, E_inf, E_hor
    
    
    Flux_Info_a0999999999.sort()
    
#    print(sum(np.array(Flux_Info_a0999999999[0][1])) + sum(np.array(Flux_Info_a0999999999[0][2])))

#    return Flux_Info_a0999999999

#No_Files = len(os.listdir('/Users/Ollie/Documents/Thesis/NHEK_Geometry/NHEK_Code/Gargantua_data/a0.999999999'))
#Flux_Info_Ext = []
#for i in range(0,No_Files):
#    item = os.listdir('/Users/Ollie/Documents/Thesis/NHEK_Geometry/NHEK_Code/Gargantua_data/a0.999999999')[i]
#    if item == ".DS_Store" or item == "ReadFluxes_a0999999999.py" or item == '__pycache__':
#        continue
#    file_name_flux = item
#    r = float(re.findall("\d+\.\d+",file_name_flux)[0])     # Radii the flux is calculated at 
#
#    E_Inf_mode = [float(column.split(' ')[2]) for column in open(str(file_name_flux)).readlines()]
#    E_Hor_mode = [float(column.split(' ')[3]) for column in open(str(file_name_flux)).readlines()]
#                                    # ^ The above reads in the data as columns.
#    
#
#    E_Inf_Modes = []
#    E_Hor_Modes = []
#    
#    
#
#    for l in range(1,len(E_Inf_mode)):
#        E_Inf_Modes.append(sum(E_Inf_mode[0:(l+1)]))  # Find the sum of the Inf modes up to m <= l
#        E_Hor_Modes.append(sum(E_Hor_mode[0:(l+1)]))  # similar but for horizon.
#        del E_Hor_mode[:(l+1)]
#        del E_Inf_mode[:(l+1)]  # delete the modes.
#        
#    E_Inf_Modes = [item for item in E_Inf_Modes if item != 0]   # Remove the zeros (machine precision)
#
#    E_Hor_Modes = [item for item in E_Hor_Modes if item != 0]
#        
#    
#    Flux_Info_Ext.append([r,E_Inf_Modes,E_Hor_Modes])  # add the data in [_,_,_] for r, E_inf, E_hor
#
#
#Flux_Info_Ext.sort()
#
#
#
##    print(sum(np.array(Flux_Info_a0999999999[0][1])) + sum(np.array(Flux_Info_a0999999999[0][2])))




        
    
        

        
    
    
    
  
        


