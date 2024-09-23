import numpy as np

def exp1(t, N0, lam, C) : return N0*np.exp(-lam*t) + C

def exp2(t, N0, lam1, lam2, C) : return N0*np.exp(-lam1*t) + 1.3475*N0*np.exp(-lam2*t) + C

#If the model passes through the error bars, chi2 = N (number of data)
def chi2(data, fit, error) : return np.sum((data - fit)**2/error**2)

#For a good model, chi2_norm = 1 +/- sqrt(2/(N - N_param))
def chi2_norm(data, fit, error, N_param) : return (np.sum((data - fit)**2/error**2))/(len(data) - N_param)










