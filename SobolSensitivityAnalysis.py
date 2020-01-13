
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:24:05 2020

@author: kamilalarripa


This program applies Sobol Sensitivity Analysis to our model of 6 ODEs describing the immunomodulatory impact of
macrophage metabolism. It then creates a bar plot of the first order and total sensitivities.

It makes use of the module SALib.

Our paper considers first the IL-1beta concentration at 24 hours to be the outcome of interest, then the LPS concentration
at 24 hours to be the outcome of interest.  To run the code for the second, comment and uncomment where indicated.  The three changes will be
1. Y[i] = ,  2. plt.title text and 3. file name for pdf in f.savefig.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from SALib.sample import saltelli
from SALib.analyze import sobol



"""
Solve ODE for fixed parameters and initial conditions
"""


# solve the system dy/dt = f(y, t)

""" Hill Function, used in system of ODEs """
# upregulating hill function
def hillup(x, b, hill): # hill is for the exponent, our model uses hill = 1
  y = x**hill/(b**hill + x**hill) # from 0 to 1
  return y

""" Define state variables and parameters and system of equations """

def f(y, t, paras):

    P = y[0] # pathogen
    C = y[1] # citrate
    I = y[2] # itaconate
    S = y[3] # succinate
    IL = y[4] # IL-1Beta
    N = y[5] # nitric oxide

    try:
        # grouped by equation where parameter first appears, parameter numbers in comments
        # eq 1 
        rP = paras['rP'].value # 1
        aIP = paras['aIP'].value # 2
        bIP = paras['bIP'].value # 3
        dP = paras['dP'].value # 4
        aNP = paras['aNP'].value # 5
        bNP = paras['bNP'].value # 6
        aILP = paras['aILP'].value # 7
        bILP = paras['bILP'].value # 8
        # eq 2 
        sC = paras['sC'].value # 9
        aCI = paras['aCI'].value # 10
        bCI = paras['bCI'].value # 11
        aCS = paras['aCS'].value # 12
        bCS = paras['bCS'].value # 13
        aCN = paras['aCN'].value # 14
        bCN = paras['bCN'].value # 15
        # eq 3 
        # aCI and bCI in eq 2
        dI =paras['dI'].value # 16
        #eq 4-- aCS and bCS in eq 2 
        sS = paras['sS'].value # 17
        dS = paras['dS'].value # 18
        aNS = paras['aNS'].value # 19
        bNS = paras['bNS'].value # 20
        aIS = paras['aIS'].value # 21
        bIS = paras['bIS'].value # 22
        # eq 5 
        aPSIL = paras['aPSIL'].value # 23
        bPIL = paras['bPIL'].value # 24
        bSIL = paras['bSIL'].value # 25
        aIIL = paras['aIIL'].value # 26
        bIIL = paras['bIIL'].value # 27
        dIL = paras['dIL'].value # 
        # eq 6-- aCN and bCN in eq 2 
        dN = paras['dN'].value # 29

    except:
        rP, aIP, bIP, dP, aNP, bNP, aILP, bILP, sC, aCI, bCI, aCS, bCS, aCN, bCN, dI, sS, dS, aNS, bNS, aIS, bIS, aPSIL, bPIL, bSIL, aIIL, bIIL, dIL, dN = paras # 29 parameters
    
    # right hand side of ODE model
    f0 = rP * (1-aIP * hillup(I, bIP, 1)) * P - dP * (1 + aNP * hillup(N ,bNP, 1) + aILP * hillup(IL,bILP,1)) * P
    f1 = sC - aCI * hillup(C,bCI,1) - aCS * hillup(C,bCS,1) - aCN * hillup(C,bCN,1)
    f2 = aCI * hillup(C,bCI,1)*P -dI*I
    f3 = aCS * hillup(C,bCS,1) + sS * P - dS * (1 - aNS * hillup(N,bNS,1) - aIS * hillup(I,bIS,1)) * S
    f4 = aPSIL * hillup(P,bPIL,1) * hillup(S,bSIL,1) * (1- aIIL * hillup(I,bIIL,1)) - dIL * IL 
    f5 = aCN * hillup(C,bCN,1)*P -dN * N
    return [f0, f1, f2, f3, f4, f5]



# initial conditions from Cordes 2016 paper (for P, C, I, S)
P0 = 5.              # initial pathogen
C0 = 2.34                 # initial citrate
I0 = .10                 # initial itaconate
S0 = .60                 # intial succinate
IL0 = 0                # inital IL-1beta
N0 = 0                 # initial nitric oxide
y0 = [P0, C0, I0, S0, IL0, N0]     # initial condition vector
t = np.linspace(0, 24., 100)         # time grid, 0 to 24 hours in 100 steps 

# parameter values (fixed after parameter estimation, see baseline column in table 1 of paper).  Grouped by equation in which parameter first appears.

# eq 1
rP = 1.15
aIP = 2
bIP = 10
dP = .001 
aNP = 1.98079282
bNP = 0.00100000 
aILP = 9.87710386
bILP = 1.46175754
# eq 2 
sC = 10 
aCI =25
bCI = 200 # also in eq 3 
aCS = 10
bCS = 3.5 # also in eq 4
aCN = 6.14661990 # also in eq 6
bCN = 0.00100000 # also in eq 6
# eq 3 
# aCI and bCI in eq 2
dI = 0.001000001
# eq 4
# aCS and bCS in eq 2
sS = .8333
dS = 8 
aNS = 0.06119352 
bNS = 0.00235215
aIS = 0.00100000
bIS = 9.67344141
# eq 5
aPSIL = 0.50781571
bPIL = 0.00100000
bSIL = 0.00100000
aIIL = 1.67992108
bIIL = 0.00100000
dIL = 1 
# eq 6
dN = 1.035



# vary parameters by 15% in each direction for sensitivity analysis
problem = {
    'num_vars': 29,
    'names': ['rP', 'aIP', 'bIP', 'dP', 'aNP', 'bNP', 'aILP', 'bILP', 'sC', 'aCI', 'bCI', 'aCS', 'bCS', 'aCN', 'bCN', 'dI', 'sS', 'dS', 'aNS', 'bNS', 'aIS', 'bIS', 'aPSIL', 'bPIL', 'bSIL', 'aIIL', 'bIIL', 'dIL','dN'],
    'bounds': [[.85*rP, 1.15*rP], # rP
               [.85*aIP, 1.15*aIP], # aIP
               [.85*bIP, 1.15*bIP], # bIP
               [.85*dP, 1.15*dP], # dP
               [.85*aNP, 1.15*aNP], # aNP
               [.85*bNP, 1.15*bNP], # bNP
               [.85*aILP, 1.15*aILP], # aILP
               [.85*bILP, 1.15*bILP], # bILP
               [.85*sC, 1.15*sC], # sC
               [.85*aCI, 1.15*aCI], # aCI 10
               [.85*bCI, 1.15*bCI],# bCI = 200
               [.85*aCS, 1.15*aCS], # aCS
               [.85*bCS, 1.15*bCS], # bCS
               [.85*aCN, 1.15*aCN], # aCN
               [.85*bCN, 1.15*bCN], # bCN
               [.85*dI, 1.15*dI], # dI 16
               [.85*sS, 1.15*sS], # sS
               [.85*dS, 1.15*dS], # dS
               [.85*aNS, 1.15*aNS], # aNS
               [.85*bNS, 1.15*bNS], # bNS
               [.85*aIS, 1.15*aIS], # aIS
               [.85*bIS, 1.15*bIS], # bIS
               [.85*aPSIL, 1.15*aPSIL], # aPSIL
               [.85*bPIL, 1.15*bPIL], # bPIL
               [.85*bSIL, 1.15*bSIL], # bSIL
               [.85*aIIL, 1.15*aIIL], # aIIL
               [.85*bIIL, 1.15*bIIL], # bIIL
               [.85*dIL, 1.15*dIL], # dILB
               [.85*dN, 1.15*dN]] # dN
}

# generate samples
param_values = saltelli.sample(problem, 10000) # N = 1000 gives 60,000 rows, 29 columns, one column for each parameter: rP,aIP... for 29 parameters, N = 10,000 gives 600,000 samples
# to see dimensions use param_values.shape
Y = np.zeros([param_values.shape[0]]) # make array to hold solution at last time step for pathogen or LPS (outcome of interest)

for i in range(param_values.shape[0]): # iteratres through rows of param_values array, moving through parameter samples
    
    
    rP = param_values[i,0]
    aIP = param_values[i,1]
    bIP = param_values[i,2]
    dP = param_values[i,3]
    aNP = param_values[i,4]
    bNP = param_values[i,5]
    aILP = param_values[i,6]
    bILP = param_values[i,7]
    sC = param_values[i,8]
    aCI = param_values[i,9]
    bCI = param_values[i,10]
    aCS = param_values[i,11]
    bCS = param_values[i,12]
    aCN = param_values[i,13]
    bCN = param_values[i,14]
    dI = param_values[i,15]
    sS = param_values[i,16]
    dS = param_values[i,17]
    aNS = param_values[i,18]
    bNS = param_values[i,19]
    aIS = param_values[i,20]
    bIS = param_values[i,21]
    aPSIL = param_values[i,22]
    bPIL = param_values[i,23]
    bSIL = param_values[i,24]
    aIIL = param_values[i,25]
    bIIL = param_values[i,26]
    dILB = param_values[i,27]
    dN = param_values[i,28]
    

    # solve system of ODEs
    soln = odeint(f, y0, t, args=((rP, aIP, bIP, dP, aNP, bNP, aILP, bILP, sC, aCI, bCI, aCS, bCS, aCN, bCN, dI, sS, dS, aNS, bNS, aIS, bIS, aPSIL, bPIL, bSIL, aIIL, bIIL, dIL, dN), )) 
    
    # extract each state variable, and store solutuion in its own matrix
    P = soln[:, 0] # pathogen
    C = soln[:, 1] # citrate
    I = soln[:, 2] # itaconate
    S = soln[:, 3] # succinate
    IL = soln[:, 4] # IL-1Beta
    N = soln[:, 5] # nitric oxide
    
 #  let P[-1], pathogen at last time step be the outcome of interest
    #Y[i] = P[-1]
    
     # let IL[-1], ILB at last time step be the outcome of interest
    Y[i] = IL[-1]
    
    
# perform sensitivity analysis
    
# sobol.analyze, computes first, second, and total-order indices.
# Si is a Python dict with the keys "S1", "S2", "ST", "S1_conf", "S2_conf", and "ST_conf

Si = sobol.analyze(problem, Y)


# to access and understand the dictionary
 # Si.keys() returns dict_keys(['S1', 'S1_conf', 'ST', 'ST_conf', 'S2', 'S2_conf'])
 # Si.get('S1')  returns array with values for S1
 
# print interactions to console
print(Si['S1'])  
print(Si['ST'])
# if the values for ST are large, it is likely that there are higher order interactions occuring.

"""

plot bar graphs with error bars

"""
        
 # Si.get('S1') # returns array with values for S1    
A = Si.get('S1')
# change array to list for plotting
S1_values = A.tolist() # list of S1 values
# get ST values
B = Si.get('ST')
# change array to list for plotting
ST_values = B.tolist()
 # get confidence intervals for ST
C = Si.get('ST_conf')
# change array to list for plotting
STconf_values = C.tolist()
# get confidence intervals for S1
D = Si.get('S1_conf')
# change array to list for plotting
S1conf_values = D.tolist()
 
 # width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars-- ST
bars1 = ST_values
 
# Choose the height of the cyan bars-- S1
bars2 = S1_values
 
# Choose the height of the error bars (bars1)-- CI for ST
yer1 = STconf_values
# Choose the height of the error bars (bars2)-- CI for S1
yer2 = S1conf_values
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

f = plt.figure()
 
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', yerr=yer1, capsize=7, label='Total Sensitivity')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', yerr=yer2, capsize=7, label='1st order Sensitivity')
 
## general layout
plt.xticks([r + barWidth for r in range(len(bars1))], ['rP', 'aIP', 'bIP','dP','aNP','bNP','aILP','bILP','sC','aCI','bCI', 'aCS', 'bCS', 'aCN', 'bCN', 'dI', 'sS', 'dS', 'aNS', 'bNS', 'aIS', 'bIS', 'aPSIL', 'bPIL', 'bSIL', 'aIIL', 'bIIL', 'dIL','dN'],rotation='vertical')
plt.ylabel('Sensitivity Index')
plt.legend()
plt.title('Sensitivity with Respect to IL-1Beta Concentration at 24 hours')
#plt.title('Sensitivity with Respect to LPS Concentration at 24 hours')

# view and save figure
plt.show()
#f.savefig("SensitivityfigLPS.pdf",bbox='tight')
f.savefig("SensitivityfigILB.pdf",bbox='tight')


