#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:43:00 2019

@author: Ollie
"""
import os
import re

def ReadAllData_a5():
    path_a09995,path_a4,path_a5,path_Ext,path_a0998,path_a097,Home_Of_Code = Paths()
    No_Files = len(os.listdir(path_a5))
    Flux_Info_a5 = []
    
    for i in range(0,No_Files):
        item = os.listdir(path_a5)[i]
        if item == ".DS_Store" or item == "ReadAllData_a5.py" or item == '__pycache__':
            continue
    
        r = float(re.findall("-?\d+\.?\d*",item)[-1])    # Radii the flux is calculated at 
    
        E_Inf_mode = [float(column.split(' ')[2]) for column in open(str(item)).readlines()]
        E_Hor_mode = [float(column.split(' ')[3]) for column in open(str(item)).readlines()]
        
        E_Tot_Inf = sum(E_Inf_mode)
        E_Tot_Hor = sum(E_Hor_mode)
        
        E_GW = (E_Tot_Inf + E_Tot_Hor)
    
        E_Inf_Modes = []
        E_Hor_Modes = []
    
        for l in range(1,len(E_Inf_mode)):
            E_Inf_Modes.append(sum(E_Inf_mode[0:(l+1)]))
            E_Hor_Modes.append(sum(E_Hor_mode[0:(l+1)]))
            del E_Hor_mode[:(l+1)]
            del E_Inf_mode[:(l+1)]
            
        E_Inf_Modes = [item for item in E_Inf_Modes if item != 0]
    
        E_Hor_Modes = [item for item in E_Hor_Modes if item != 0]
            
        
        Flux_Info_a5.append([r,E_Inf_Modes,E_Hor_Modes])
    
    Flux_Info_a5.sort()
    
    r_a5,Flux_a5_GW = [],[]
    for i in range(0,len(Flux_Info_a5)):
        r_a5.append(Flux_Info_a5[i][0])  # Radial coordinates
        Flux_a5_GW.append(sum(Flux_Info_a5[i][1]) + sum(Flux_Info_a5[i][2]))  # Total energy flux
    
    Eps_a5 = [(5/32) * 1 * ((r_a5[i])**(3/2) + (1-10**-5))**(10/3)*Flux_a5_GW[i] for i in range(0,len(Flux_Info_a5))]

    return Flux_Info_a5

