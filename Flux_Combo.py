#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:58:48 2019

@author: Ollie
"""

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
    path_a09995,path_a4,path_a5,path_Ext,path_a0998,path_a097,Home_Of_Code = Paths()
    os.chdir(path_Ext)
    No_Files = len(os.listdir(path_Ext))
    Flux_Info_a0999999999 = []
    for i in range(0,No_Files):
        item = os.listdir(path_Ext)[i]
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
    Check = []
    
    for i in range(0,len(Flux_Info_a0999999999)):
        r = Flux_Info_a0999999999[i][0]
        E_inf = Flux_Info_a0999999999[i][1][0:14]
        E_hor = Flux_Info_a0999999999[i][2][0:14]
        Check.append([r,E_inf,E_hor])
    Flux_Info_a0999999999 = Check

    return Flux_Info_a0999999999



def ReadAllData_a5():
    path_a09995,path_a4,path_a5,path_Ext,path_a0998,path_a097,Home_Of_Code = Paths()
    os.chdir(path_a5)
    No_Files = len(os.listdir(path_a5))
    Flux_Info_a5 = []
    
    for i in range(0,No_Files):
        item = os.listdir(path_a5)[i]
        if item == ".DS_Store" or item == "FluxData_a5.py" or item == '__pycache__':
            continue
    
        r = float(re.findall("-?\d+\.?\d*",item)[-1])    # Radii the flux is calculated at 
    
        E_Inf_mode = [float(column.split(' ')[2]) for column in open(str(item)).readlines()]
        E_Hor_mode = [float(column.split(' ')[3]) for column in open(str(item)).readlines()]
        
#        E_Tot_Inf = sum(E_Inf_mode)
#        E_Tot_Hor = sum(E_Hor_mode)
#        
#        E_GW = (E_Tot_Inf + E_Tot_Hor)
    
        E_Inf_Modes = []
        E_Hor_Modes = []
        # modes
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



def ReadAllData_a4():
    path_a09995,path_a4,path_a5,path_Ext,path_a0998,path_a097,Home_Of_Code = Paths()
    os.chdir(path_a4)
    No_Files = len(os.listdir(path_a4))
    Flux_Info_a4 = []
    
    for i in range(0,No_Files):
        item = os.listdir(path_a4)[i]
        if item == ".DS_Store" or item == "FluxData_a4.py" or item == '__pycache__':
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
            
        
        Flux_Info_a4.append([r,E_Inf_Modes,E_Hor_Modes])
    
    Flux_Info_a4.sort()
    return Flux_Info_a4


def ReadAllData_a09995():
    path_a09995,path_a4,path_a5,path_Ext,path_a0998,path_a097,Home_Of_Code = Paths()
    os.chdir(path_a09995)
    No_Files = len(os.listdir(path_a09995))
    Flux_Info_a09995 = []
    
    for i in range(0,No_Files):
        item = os.listdir(path_a09995)[i]
        if item == ".DS_Store" or item == "FluxData_a09995.py" or item == '__pycache__':
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
            
        
        Flux_Info_a09995.append([r,E_Inf_Modes,E_Hor_Modes])
    
    Flux_Info_a09995.sort()
    
    return Flux_Info_a09995

    
def ReadAllData_a0998():
    path_a09995,path_a4,path_a5,path_Ext,path_a0998,path_a097,Home_Of_Code = Paths()
    os.chdir(path_a0998)
    No_Files = len(os.listdir(path_a0998))
    Flux_Info_a0998 = []
    for i in range(0,No_Files):
        item = os.listdir(path_a0998)[i]
        if item == ".DS_Store" or item == "FluxData_a0998.py" or item == '__pycache__':
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
            
        
        Flux_Info_a0998.append([r,E_Inf_Modes,E_Hor_Modes])

    Flux_Info_a0998.sort()
    return Flux_Info_a0998

def ReadAllData_a097():
    path_a09995,path_a4,path_a5,path_Ext,path_a0998,path_a097,Home_Of_Code = Paths()
    os.chdir(path_a097)    
    No_Files = len(os.listdir(path_a097))
    Flux_Info_a097 = []
    for i in range(0,No_Files):
        item = os.listdir(path_a097)[i]
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




        
    
        

        
    
    
    
  
        


