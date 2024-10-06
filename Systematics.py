import numpy as np
import BigFunctions as BF
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import uproot as ur
import warnings
import gc
warnings.filterwarnings('ignore')


#Data
#Without concatenation
"""
input_path = "../data/bigdata_CFD_26092024_0001.root"
with ur.open(input_path) as file:
    outTree = file["DataTree"]
    label = outTree["label"].array(library = "np")
    time = outTree["time"].array(library = "np")*10**-6   #[µs]
    nrj = outTree["nrj"].array(library = "np")
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

def Analyse(corr_t, b, dt_max, N_bins) :
    print("########Analysis for\ncorr_t =", corr_t*10**3, '[ns]\nb =', b, '\ndt_max =', dt_max, '[µs]\nN_bins =', N_bins)

    #Time Correlation
    corr_mask = (time[1:]-time[:-1]<corr_t) & (label[1:] != label[:-1])
    correl_both = (np.insert(corr_mask,0,False))|(np.append(corr_mask,False))
    del corr_mask
    t1 = time[correl_both & (label==1)]
    t2 = time[correl_both & (label==2)]
    nrj1 = nrj[correl_both & (label==1)]
    nrj2 = nrj[correl_both & (label==2)]
    dt2 = t2[1:]-t2[:-1]

    #Energy Correlation
    nrj1_max = 0.5e6
    nrj2_max = 1e6
    mask1 = (nrj1 < nrj1_max)&(nrj2 < nrj2_max)
    arr_1,xe_1,ye_1 = np.histogram2d(nrj1[mask1], 
                                     nrj2[mask1], 
                                     bins =(100,100))
    popt, cov = curve_fit(BF.lin, xe_1, ye_1)
    a = popt[0]
    mask2 = (a*nrj1+b>=nrj2) & (a*nrj1-b <= nrj2)
    nrj22 = nrj2[mask1 & mask2]
    t22 = t2[mask1 & mask2]
    dt22 = t22[1:]-t22[:-1]
    del nrj1
    del nrj2
    del t1

    #Additional Cuts
    dt = dt22[(dt22 < dt_max)&(nrj22[:-1]<0.25e6)&(nrj22[1:]<0.25e6)]
    nrjµ = nrj22[:-1][(dt22 < dt_max)&(nrj22[:-1]<0.25e6)&(nrj22[1:]<0.25e6)]
    nrje = nrj22[1:][(dt22 < dt_max)&(nrj22[:-1]<0.25e6)&(nrj22[1:]<0.25e6)]
    dt_final = dt22[(dt22 < dt_max)&(nrj22[:-1]<0.25e6)&(nrj22[1:]<50000)]
    del dt
    del dt22
    del nrj22

    #Fit
    hist = np.histogram(dt_final, bins = N_bins)
    x = hist[1][1:]-(hist[1][1]-hist[1][0])/2
    y = hist[0]
    y_err = np.sqrt(y)
    popt, cov = curve_fit(BF.exp4, x, y, sigma = y_err, absolute_sigma = True, p0 = np.array([4e6, 2.2, 0.1, 200]))
    N0 = popt[0]
    tau= popt[1]
    tau_err = np.sqrt(cov[1,1])
    lambda_c = popt[2]
    lambda_c_err = np.sqrt(cov[2,2])
    C = popt[3]
    chi2 = BF.chi2_norm(y, BF.exp4(x, N0, tau, lambda_c, C), y_err, 4)[0]
    chi2_err = BF.chi2_norm(y, BF.exp4(x, N0, tau, lambda_c, C), y_err, 4)[1]
    del hist
    gc.collect()
    return tau, tau_err, lambda_c, lambda_c_err, chi2, chi2_err

#Parameters to vary
#corr_t = 5*10**-3   #Time correlation window
corr_t_lst = np.arange(5, 51)*10**-3

b = 25000   #Point at origin for energy correlation
#b_lst = np.linspace(0.005*10**6, 0.05*10**6, 100)
dt_max = 30   #Maximum delta t

N_bins = 200   #Number of bins used for the fit
#N_bins_lst = np.linspace(50, 300, 51)


#corr_t variation
tau_corr_t = []
lambda_c_corr_t = []
fig, ax = plt.subplots(2, 1, sharex=True)
for corr_t in corr_t_lst :
    tau, tau_err, lambda_c, lambda_c_err, chi2, chi2_err = Analyse(corr_t, b, dt_max, N_bins)
    tau_corr_t.append(tau)
    lambda_c_corr_t.append(lambda_c)
    ax[0].scatter(corr_t*10**3, tau, c = 'k')
    ax[0].errorbar(corr_t*10**3, tau, yerr = tau_err, linestyle = 'None', capsize=4, c='k')
    ax[1].scatter(corr_t*10**3, chi2, c = 'k')
    ax[1].errorbar(corr_t*10**3, chi2, yerr = chi2_err, linestyle = 'None', capsize=4, c='k')
plt.xlabel(r'$t_{corr}$ [ns]')
ax[0].set_ylabel(r'$\tau$ [µs]')
ax[1].set_ylabel(r'$\chi^2$/ndf')
ax[0].grid()
ax[1].grid()
plt.savefig('corr_t_2')
print("Uncertainty on tau due to corr_t variation", np.std(np.array(tau_corr_t)), '[µs]')
del tau_corr_t
print("Uncertainty on lambda_c due to corr_t variation", np.std(np.array(lambda_c_corr_t)), '[µs-1]')
del lambda_c_corr_t

corr_t = 5*10**-3   #Time correlation window

#b variation
tau_b = []
lambda_c_b = []
b_lst = np.linspace(0.005*10**6, 0.05*10**6, 91)
fig, ax = plt.subplots(2, 1, sharex=True)
for b in b_lst :
    tau, tau_err, lambda_c, lambda_c_err, chi2, chi2_err = Analyse(corr_t, b, dt_max, N_bins)
    tau_b.append(tau)
    lambda_c_b.append(lambda_c)
    ax[0].scatter(b, tau, c = 'k')
    ax[0].errorbar(b, tau, yerr = tau_err, linestyle = 'None', capsize=4, c='k')
    ax[1].scatter(b, chi2, c = 'k')
    ax[1].errorbar(b, chi2, yerr = chi2_err, linestyle = 'None', capsize=4, c='k')
plt.xlabel(r'$b$ [arbitrary unit]')
ax[0].set_ylabel(r'$\tau$ [µs]')
ax[1].set_ylabel(r'$\chi^2$/ndf')
ax[0].grid()
ax[1].grid()
plt.savefig('b_2')
print("Uncertainty on tau due to b variation", np.std(np.array(tau_b)), '[µs]')
del tau_b
print("Uncertainty on lambda_c due to b variation", np.std(np.array(lambda_c_b)), '[µs-1]')
del lambda_c_b

b=25000

#dt_max variation
tau_dt_max = []
lambda_c_dt_max = []
dt_max_lst = np.arange(10, 101)
fig, ax = plt.subplots(2, 1, sharex=True)
for dt_max in dt_max_lst :
    tau, tau_err, lambda_c, lambda_c_err, chi2, chi2_err = Analyse(corr_t, b, dt_max, N_bins)
    tau_dt_max.append(tau)
    lambda_c_dt_max.append(lambda_c)
    ax[0].scatter(dt_max, tau, c = 'k')
    ax[0].errorbar(dt_max, tau, yerr = tau_err, linestyle = 'None', capsize=4, c='k')
    ax[1].scatter(dt_max, chi2, c = 'k')
    ax[1].errorbar(dt_max, chi2, yerr = chi2_err, linestyle = 'None', capsize=4, c='k')
plt.xlabel(r'$dt_{max}$ [µs]')
ax[0].set_ylabel(r'$\tau$ [µs]')
ax[1].set_ylabel(r'$\chi^2$/ndf')
ax[0].grid()
ax[1].grid()
plt.savefig('dt_max_2')
print("Uncertainty on tau due to dt_max variation", np.std(np.array(tau_dt_max)), '[µs]')
del tau_dt_max
print("Uncertainty on lambda_c due to dt_max variation", np.std(np.array(lambda_c_dt_max)), '[µs-1]')
del lambda_c_dt_max

dt_max = 30

#N_bins variation
tau_N_bins = []
lambda_c_N_bins = []
N_bins_lst = np.linspace(50, 300, 51)
fig, ax = plt.subplots(2, 1, sharex=True)
for N_bins in N_bins_lst :
    tau, tau_err, lambda_c, lambda_c_err, chi2, chi2_err = Analyse(corr_t, b, dt_max, round(N_bins))
    tau_N_bins.append(tau)
    lambda_c_N_bins.append(lambda_c)
    ax[0].scatter(N_bins, tau, c = 'k')
    ax[0].errorbar(N_bins, tau, yerr = tau_err, linestyle = 'None', capsize=4, c='k')
    ax[1].scatter(N_bins, chi2, c = 'k')
    ax[1].errorbar(N_bins, chi2, yerr = chi2_err, linestyle = 'None', capsize=4, c='k')
plt.xlabel(r'$N_{bins}$')
ax[0].set_ylabel(r'$\tau$ [µs]')
ax[1].set_ylabel(r'$\chi^2$/ndf')
ax[0].grid()
ax[1].grid()
plt.savefig('N_bins_2')
print("Uncertainty on tau due to N_bins variation", np.std(np.array(tau_N_bins)), '[µs]')
del tau_N_bins
print("Uncertainty on lambda_c due to N_bins variation", np.std(np.array(lambda_c_N_bins)), '[µs-1]')
del lambda_c_N_bins

