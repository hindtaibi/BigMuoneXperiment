import numpy as np

def exp1(t, N0, lam, C) : return N0*np.exp(-lam*t) + C

def exp2(t, N0, lam1, lam2, C) : return N0*np.exp(-lam1*t) + 1.3475*N0*np.exp(-lam2*t) + C













