import numpy as np

def exp2(t, N0, lam) : return N0*np.exp(-lam*t)

def exp3(t, N0, lam, C) : return N0*np.exp(-lam*t) + C

#def exp4(t, N0, lam1, lam2, C) : return N0*np.exp(-lam1*t) + 1.2*N0*np.exp(-lam2*t) + C

def exp5(t, N01, N02, lam1, lam2, C) : return N01*np.exp(-lam1*t) + N02*np.exp(-lam2*t) + C

def exp4(t, N0, tau, lambda_c, C) : return N0*np.exp(-t/tau)*(1.2 + np.exp(-lambda_c*t)) + C

#If the model passes through the error bars, chi2 = N (number of data)
def chi2(data, fit, error) : 
    return np.sum((data - fit)**2/error**2)

#For a good model, chi2_norm = 1 +/- sqrt(2/(N - N_param))
def chi2_norm(data, fit, error, N_param) : 
    return (np.sum((data - fit)**2/error**2))/(len(data) - N_param), np.sqrt(2/(len(data) - N_param))

def lin(x, a) : return a*x

def affine(x, a, b) : return a*x+b










