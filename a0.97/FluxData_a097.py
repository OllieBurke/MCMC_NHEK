#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:03:12 2018

@author: Ollie
"""

def ReadAllData_a097():
    No_Files = len(os.listdir('/Users/Ollie/Documents/Thesis/NHEK_Geometry/NHEK_Code/Gargantua_data/a0.97'))
    Flux_Info_a097 = []
    for i in range(0,No_Files):
        item = os.listdir('/Users/Ollie/Documents/Thesis/NHEK_Geometry/NHEK_Code/Gargantua_data/a0.97')[i]
        if item == ".DS_Store" or item == "FluxData_a097.py" or item == '__pycache__':
            continue
        r = float(re.findall("\d+\.\d+",item)[0])    # Radii the flux is calculated at 
        
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
            
        
        Flux_Info_a097.append([r,E_Inf_Modes,E_Hor_Modes])

    Flux_Info_a097.sort()
    return Flux_Info_a097