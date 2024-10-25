import numpy as np
import BigFunctions as BF
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import uproot as ur
import warnings
import gc
warnings.filterwarnings('ignore')

print('####################### Parameters #######################\n')

corr_t = 5*10**-3   #Time correlation window
b = 25000   #Point at origin for energy correlation
dt_max = 30   #Maximum delta t
N_bins = 200   #Number of bins used for the fit

print("Time correlation window:               ", corr_t*10**3, "[ns]")
print("Point at origin for energy corrleation:", b)
print("Maximum delta t:                       ", dt_max, "[µs]")
print("Number of bins used for the fit:       ", N_bins)

print('\n##########################################################\n')

print('#################### Reading the Data ####################\n')

#Without concatenation
"""
input_path = "../data/bigdata_CFD_26092024_0001.root"
with ur.open(input_path) as file:
    outTree = file["DataTree"]
    label = outTree["label"].array(library = "np")
    time = outTree["time"].array(library = "np")*10**-6   #[µs]
    nrj = outTree["nrj"].array(library = "np")
    pileup = outTree["pileup"].array(library = "np")

print("Duration of the data acquisition:  ", max(time)*10**(-6)/3600, "[hours]")
print("Total number of pileups:           ", len(pileup[pileup == True]))
print("Number of events acquired:         ", len(label)/10**6, "[million] (100 %)")
print("Number of events acquired with PM2:", len(label[label == 2])/10**6, "[million] (100 %)")
"""

#With concatenation
def vars_data(data):
    with ur.open(data) as file:
        outTree = file["DataTree"]
        label = outTree["label"].array(library = "np")
        time = outTree["time"].array(library = "np")*10**-6   #[µs]
        nrj = outTree["nrj"].array(library = "np")
        pileup = outTree["pileup"].array(library = "np")    
    return label, time, nrj, pileup

vars_23_09 = vars_data("../data/bigdata_CFD_23092024_0001.root")
vars_26_09 = vars_data("../data/bigdata_CFD_26092024_0001.root")

label = np.concatenate((vars_23_09[0], vars_26_09[0]), axis=None)
time = np.concatenate((vars_23_09[1], 100+vars_23_09[1][-1]+vars_26_09[1]), axis=None) 
nrj = np.concatenate((vars_23_09[2], vars_26_09[2]), axis=None)
pileup = np.concatenate((vars_23_09[3], vars_26_09[3]), axis=None)

print("Duration of the data acquisition:  ", max(time)*10**(-6)/3600, "[hours]")
print("Total number of pileups:           ", len(pileup[pileup == True]))
print("Number of events acquired:         ", len(label)/10**6, "[million] (100 %)")
print("Number of events acquired with PM2:", len(label[label == 2])/10**6, "[million] (100 %)")

print('\n##########################################################\n')

print('#################### Time Correlation ####################\n')

corr_mask = (time[1:]-time[:-1]<corr_t) & (label[1:] != label[:-1])

correl_both = (np.insert(corr_mask,0,False))|(np.append(corr_mask,False))
del(corr_mask)

t = time[correl_both]
l = label[correl_both]
p = pileup[correl_both]
e = nrj[correl_both]

t1 = time[correl_both & (label==1)]
t2 = time[correl_both & (label==2)]
nrj1 = nrj[correl_both & (label==1)]
nrj2 = nrj[correl_both & (label==2)]

dt2 = t2[1:]-t2[:-1]
dt1 = t1[1:]-t1[:-1]

print("Number of events after correlation:                       ", len(t)/10**6, "million")
print("Dead time:                                                ", min(dt2), "µs")
print("Total number of pileups after correlation:                ", len(p[p == True]))
print("Number of events acquired with PM2 after time correlation:", len(l[l == 2])/10**6, "million (", len(l[l == 2])*100/len(label[label == 2]), "% )")

del t
del p
del e
del time

print('\n##########################################################\n')

print('################### Energy Correlation ###################\n')

nrj1_max = 0.5e6
nrj2_max = 1e6
mask1 = (nrj1 < nrj1_max)&(nrj2 < nrj2_max)

#Linear Fit
arr_1,xe_1,ye_1 = np.histogram2d(nrj1[mask1], 
                                 nrj2[mask1], 
                                 bins =(100,100))

popt, cov = curve_fit(BF.lin, xe_1, ye_1)
a = popt[0]

#Energy correlation
mask2 = (a*nrj1+b>=nrj2) & (a*nrj1-b <= nrj2)
nrj22 = nrj2[mask1 & mask2]
t22 = t2[mask1 & mask2]
dt22 = t22[1:]-t22[:-1]

del nrj1
del nrj2

print("Number of events acquired with PM2 after time and energy correlation:\n", len(t22)/10**6, "[million] (", len(t22)*100/len(label[label == 2]), "% )")

del t22

print('\n##########################################################\n')

print('##################### Additional Cuts ####################\n')

dt = dt22[(dt22 < dt_max)&(nrj22[:-1]<0.25e6)&(nrj22[1:]<0.25e6)]
nrjµ = nrj22[:-1][(dt22 < dt_max)&(nrj22[:-1]<0.25e6)&(nrj22[1:]<0.25e6)]
nrje = nrj22[1:][(dt22 < dt_max)&(nrj22[:-1]<0.25e6)&(nrj22[1:]<0.25e6)]

print("Number of events acquired with PM2 after time and energy correlation, and cut on dt_max:\n", 
      len(dt)/10**6, "million (", len(dt)*100/len(label[label == 2]), "% )")

dt_final = dt22[(dt22 < dt_max)&(nrj22[:-1]<0.25e6)&(nrj22[1:]<50000)]

print("Number of events acquired with PM2 after time and energy correlation, cut on dt_max and on maximum second event energy:\n", 
      len(dt_final)/10**6, "million (", len(dt_final)*100/len(label[label == 2]), "% )")

print('\n##########################################################\n')

print('########################## Fit ###########################\n')

hist = np.histogram(dt_final, bins = N_bins)
x = hist[1][1:]-(hist[1][1]-hist[1][0])/2
y = hist[0]
y_err = np.sqrt(y)

#Without fixing lambda_c
popt, cov = curve_fit(BF.exp4, x, y, sigma = y_err, absolute_sigma = True, p0 = np.array([4e6, 2.2, 0.1, 200]))
N0 = popt[0]
tau= popt[1]
tau_err = np.sqrt(cov[1,1])
lambda_c = popt[2]
lambda_c_err = np.sqrt(cov[2,2])
C = popt[3]

chi2 = BF.chi2_norm(y, BF.exp4(x, N0, tau, lambda_c, C), y_err, 4)[0]
chi2_err = BF.chi2_norm(y, BF.exp4(x, N0, tau, lambda_c, C), y_err, 4)[1]
print("... Fit with two exponentials ...")
print("chi2 = ", chi2, '+/-', chi2_err)
print('tau = ', tau, '+/-', tau_err, '[µs]')
print('lambda_c = ', lambda_c, '+/-', lambda_c_err, '[µs]')

"""
#Fixind lambda_c
popt, cov = curve_fit(BF.exp4, x, y, sigma = y_err, absolute_sigma = True, p0 = np.array([4e6, 2.2, 200]))
N0 = popt[0]
tau= popt[1]
tau_err = np.sqrt(cov[1,1])
C = popt[2]
lambda_c = 0.0974
lambda_c_err = 0.0031

chi2 = BF.chi2_norm(y, BF.exp4(x, N0, tau, C), y_err, 3)[0]
chi2_err = BF.chi2_norm(y, BF.exp4(x, N0, tau, C), y_err, 3)[1]
print("... Fit with two exponentials ...")
print("chi2 = ", chi2, '+/-', chi2_err)
print('tau = ', tau, '+/-', tau_err, '[µs]')
print('lambda_c = ', lambda_c, '+/-', lambda_c_err, '[µs]')
"""

print('\n##########################################################\n')

del hist
gc.collect()
