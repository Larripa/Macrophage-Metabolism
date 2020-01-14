"""
Created on Tue Dec 24 14:28:49 2019

@author: kamilalarripa
"""


"""
This code uses time series data from Cordes et al (2016) to estimate parameters in our model.  Our model has 29 parameters and 6 initial conditions.
Some parameter values were obtained from the literature and held fixed, others were estimated from the data.  Estimate parameter values are printed to the console.
Graph of data points and solution is displayed in console and saved as file Fit.pdf.
Parameter estimation is performed with least squares.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit


""" Hill Function """
# upregulating hill function
def hillup(x, b, hill): # hill is for the exponent, our model uses hill = 1
  y = x**hill/(b**hill + x**hill) # from 0 to 1
  return y

"""
Solve ODE for fixed parameters and initial conditions, and graph result
"""

# solve the system dy/dt = f(y, t)
def f(y, t, paras):

    P = y[0] # pathogen
    C = y[1] # citrate
    I = y[2] # itaconate
    S = y[3] # succinate
    IL = y[4] # IL-1Beta
    N = y[5] # nitric oxide

    try:
        
        # eq 1 (8)
        rP = paras['rP'].value
        aIP = paras['aIP'].value
        bIP = paras['bIP'].value
        dP = paras['dP'].value
        aNP = paras['aNP'].value
        bNP = paras['bNP'].value
        aILP = paras['aILP'].value
        bILP = paras['bILP'].value
        # eq 2 (7)
        sC = paras['sC'].value
        aCI = paras['aCI'].value
        bCI = paras['bCI'].value
        aCS = paras['aCS'].value
        bCS = paras['bCS'].value
        aCN = paras['aCN'].value
        bCN = paras['bCN'].value
        # eq 3 (1)
        # aCI and bCI in eq 2
        dI =paras['dI'].value
        #eq 4-- aCS and bCS in eq 2 (6)
        sS = paras['sS'].value
        dS = paras['dS'].value
        aNS = paras['aNS'].value
        bNS = paras['bNS'].value
        aIS = paras['aIS'].value
        bIS = paras['bIS'].value
        # eq 5 (6)
        aPSIL = paras['aPSIL'].value
        bPIL = paras['bPIL'].value
        bSIL = paras['bSIL'].value
        aIIL = paras['aIIL'].value
        bIIL = paras['bIIL'].value
        dILB = paras['dILB'].value
        # eq 6-- aCN and bCN in eq 2 (1)
        dN = paras['dN'].value

    except:
        rP, aIP, bIP, dP, aNP, bNP, aILP, bILP, sC, aCI, bCI, aCS, bCS, aCN, bCN, dI, sS, dS, aNS, bNS, aIS, bIS, aPSIL, bPIL, bSIL, aIIL, bIIL, dILB, dN = paras # 29 parameters
    # the model equations (RHS of ODEs)
    f0 = rP * (1-aIP * hillup(I, bIP, 1)) * P - dP * (1 + aNP * hillup(N ,bNP, 1) + aILP * hillup(IL,bILP,1)) * P
    f1 = sC - aCI * hillup(C,bCI,1) - aCS * hillup(C,bCS,1) - aCN * hillup(C,bCN,1)
    f2 = aCI * hillup(C,bCI,1) * P -dI * I
    f3 = aCS * hillup(C,bCS,1) + sS * P - dS * (1 - aNS * hillup(N,bNS,1) - aIS * hillup(I,bIS,1)) * S
    f4 = aPSIL * hillup(P,bPIL,1) * hillup(S,bSIL,1) * (1- aIIL * hillup(I,bIIL,1))-dILB * IL 
    f5 = aCN * hillup(C,bCN,1) * P -dN * N
    return [f0, f1, f2, f3, f4, f5]


def g(t, x0, paras):
    """
    Solution to the ODE x'(t) = f(t,x,p) with initial condition x(0) = x0
    """
    x = odeint(f, x0, t, args=(paras,))
    return x


def residual(paras, t, data1, data2, data3):
    x0 = paras['P0'].value, paras['C0'].value, paras['I0'].value, paras['S0'].value, paras['IL0'].value, paras['N0'].value
    model = g(t, x0, paras)
    # note-- ravel returns a flattened array

    s_model = model[:, 3] # succinate is the fourth variable, so index is 3
    i_model = model[:, 2] #itaconate is third variable, so index is 2
    c_model = model[:, 1] #citrate is second variable, so index is 1
    s_residuals = s_model - data1
    i_residuals = i_model - data2
    c_residuals = c_model - data3
    return np.concatenate([s_residuals,i_residuals, c_residuals]).ravel()
    

# set seed for reproducibility 
np.random.seed(1)

# initial conditions from Cordes et al 2016 paper
P0 = 5.              # initial pathogen
C0 = 2.34                 # initial citrate, from Cordes et al
I0 = .10                 # initial itaconate
S0 = .60                 # intial succinate
IL0 = 0                # inital IL-1beta
N0 = 0                 # initial nitric oxide
y0 = [P0, C0, I0, S0, IL0, N0]     # initial condition vector

# time for simulation
n = 600 # number of time steps for solution
endtime = 6 # run for 6 hours, when time series data ends
deltat = endtime/n
t = np.linspace(0, endtime, n)         # time grid, 0 to end time hours in n steps

# initial parameter values
# eq 1
rP = 1.15
aIP = 2
bIP = 10
dP = .001 
aNP = 1 
bNP = 1 
aILP = 1
bILP = 1

# eq 2 
sC = 10
aCI =25
bCI = 200 
aCS = 10
bCS = 3.5 
aCN = 1 
bCN = .0125 

# eq 3 
# aCI and bCI in eq 2
dI = .01

# eq 4
# aCS and bCS in eq 2
sS =  .8333
dS = 8
aNS = .01 
bNS = 1
aIS = .001
bIS = 1

# eq 5
aPSIL = 10
bPIL = 1
bSIL = 1
aIIL = 1
bIIL = 1
dILB = 1 

# eq 6
# aCN and bCN in eq 2
dN = 1.035

# solve the ODEs and assign solution to state variable name

soln = odeint(f, y0, t, args=((rP, aIP, bIP, dP, aNP, bNP, aILP, bILP, sC, aCI, bCI, aCS, bCS, aCN, bCN, dI, sS, dS, aNS, bNS, aIS, bIS, aPSIL, bPIL, bSIL, aIIL, bIIL, dILB, dN ), ))
P = soln[:, 0] # pathogen
C = soln[:, 1] # citrate
I = soln[:, 2] # itaconate
S = soln[:, 3] # succinate
IL = soln[:, 4] # IL-1Beta
N = soln[:, 5] # nitric oxide





# plot results with initial parameter values
plt.figure(0)
plt.plot(t, P, label='Pathogen', color = 'b')
plt.plot(t, C, label='Citrate', color = 'y')
plt.plot(t, I, label='Itaconate',color = 'g')
plt.plot(t, S, label='Succinate', color = 'r')
plt.plot(t, IL, label='IL-1Beta', color = 'm')
plt.plot(t, N, label='Nitric Oxide', color = 'k')
plt.xlabel('Hours')
plt.ylabel('Concentration in mM (Population for Pathogen)')
plt.title('Metabolic Dynamics')
plt.legend(loc=0)
plt.show()



"""
Fit ODE model to time series data for succinate, which is the fourth variable, so index is 3.
Allow parameters/IC (rP,dP,sC) to vary and hold the other ICs/parameters to be fixed
"""

# we have time series for hours 0,1,2,3,4,5,6 hours.

# time steps for which we have data
t_measured = t[0::int(n/endtime)]
# data for succinate
S_real = S[0::int(n/endtime)] # solution to ODE at these time steps-- will start at 0 and skip by n/endtime
S_measured = [.72,.66,.70,1.04,1.13, 1.61]
# data for itaconate
I_real = I[0::int(n/endtime)] # solution to ODE at these time steps
I_measured = [.12,.29,.82,2.27,3.41,5.35] 
# data for citrate
C_real = C[0::int(n/endtime)] # solution to ODE at these time steps
C_measured = [2.56,2.01,1.97,2.19,1.79,2.06]


# set parameters including bounds or fix parameters (use vary=False)
params = Parameters()
# initial conditions, hold fixed
params.add('P0', value=P0, vary=False)
params.add('C0', value=C0, vary=False)
params.add('I0', value=I0, vary=False)
params.add('S0', value=S0, vary=False)
params.add('IL0', value=IL0, vary=False)
params.add('N0', value=N0, vary=False)

# other parameters, let 16 vary
params.add('rP', value=rP, vary=False) 
params.add('aIP', value=aIP, vary=False)
params.add('bIP', value=bIP, vary=False)
params.add('dP', value=1, vary=False) 
params.add('aNP', value=aNP, min=.001, max = 10) # vary
params.add('bNP', value=bNP, min=.001, max = 10) # vary
params.add('aILP', value=aILP, min=.001, max = 10) # vary
params.add('bILP', value=bILP, min=.001, max = 10) # vary
# eq 2
params.add('sC', value=sC, vary=False) 
params.add('aCI', value=aCI, vary=False)
params.add('bCI', value=bCI, vary=False)
params.add('aCS', value=aCS, vary=False)
params.add('bCS', value=bCS, vary=False)
params.add('aCN', value=aCN, min=.001, max = 10) # vary
params.add('bCN', value=bCN, min=.001, max = 10) # vary
# eq 3
params.add('dI', value=dI, min=.001, max = 10) # vary
# eq 4
params.add('sS', value=sS, vary=False) 
params.add('dS', value=dS, vary=False) 
params.add('aNS', value=aNS, min=.001, max = 10) # vary
params.add('bNS', value=bNS, min=.001, max = 10) # vary
params.add('aIS', value=aIS, min=.001, max = 10) # vary
params.add('bIS', value=bIS, min=.001, max = 10) # vary

# eq 5
params.add('aPSIL', value=aPSIL, min=.001, max = 10) # vary
params.add('bPIL', value=bPIL, min=.001, max = 10) # vary
params.add('bSIL', value=bSIL, min=.001, max = 10) # vary
params.add('aIIL', value=aIIL, min=.001, max = 10) # vary
params.add('bIIL', value=bIIL, min=.001, max = 10) # vary
params.add('dILB', value = dILB, vary = False)

# eq 6
params.add('dN', value=dN, vary=False)


# fit model to data using least squares
# residual function is called here
result = minimize(residual, params, args=(t_measured, S_measured, I_measured, C_measured), method='leastsq')  # leastsq nelder

# display fitted statistics
report_fit(result)


data_fitted = g(t, y0, result.params) # solution with fitted parameters

# plot results of the fit
f = plt.figure()

plt.plot(t_measured, S_measured, 'o', color='r', label='measured data succinate')
plt.plot(t_measured, C_measured, 'o', color='y', label='measured data citrate')
plt.plot(t_measured, I_measured, 'o', color='g', label='measured data itaconate')

plt.plot(t, data_fitted[:, 3], '-', linewidth=2, color='red', label='succinate solution fitted to data') # change variable to plot with index in data_fitted.  3 = succinate
plt.plot(t, data_fitted[:, 1], '-', linewidth=2, color='orange', label='citrate solution fitted to data')
plt.plot(t, data_fitted[:, 2], '-', linewidth=2, color='green', label='itaconate solution fitted to data')
plt.legend()
plt.xlabel('Hours')
plt.ylabel('Concentration in mM')


plt.show()
f.savefig("Fit.pdf",bbox='tight')





