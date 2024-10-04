import numpy as np
import BigFunctions as BF
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import uproot as ur
import warnings
warnings.filterwarnings('ignore')

#Data
input_path = "../data/bigdata_CFD_26092024_0001.root"
with ur.open(input_path) as file:
    outTree = file["DataTree"]
    label = outTree["label"].array(library = "np")
    time = outTree["time"].array(library = "np")*10**-6   #[µs]
    nrj = outTree["nrj"].array(library = "np")

def Analyse(corr_t, b, dt_max, N_bins) :
    print("########Analysis for\ncorr_t =", corr_t*10**3, '[ns]\nb =', b, '\ndt_max =', dt_max, '[µs]\nN_bins =', N_bins)

    #Time Correlation
    corr_mask = (time[1:]-time[:-1]<corr_t) & (label[1:] != label[:-1])
    correl_both = (np.insert(corr_mask,0,False))|(np.append(corr_mask,False))
    del(corr_mask)
    t = time[correl_both]
    l = label[correl_both]
    e = nrj[correl_both]
    t1 = time[correl_both & (label==1)]
    t2 = time[correl_both & (label==2)]
    nrj1 = nrj[correl_both & (label==1)]
    nrj2 = nrj[correl_both & (label==2)]
    dt2 = t2[1:]-t2[:-1]
    dt1 = t1[1:]-t1[:-1]

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

    #Additional Cuts
    dt = dt22[(dt22 < 20)&(nrj22[:-1]<0.25e6)&(nrj22[1:]<0.25e6)]
    nrjµ = nrj22[:-1][(dt22 < 20)&(nrj22[:-1]<0.25e6)&(nrj22[1:]<0.25e6)]
    nrje = nrj22[1:][(dt22 < 20)&(nrj22[:-1]<0.25e6)&(nrj22[1:]<0.25e6)]
    dt_final = dt22[(dt22 < 20)&(nrj22[:-1]<0.25e6)&(nrj22[1:]<50000)]

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
    
    return tau, tau_err, chi2, chi2_err

#Parameters to vary
corr_t = 25*10**-3   #Time correlation window
b = 125000   #Point at origin for energy correlation
dt_max = 20   #Maximum delta t
#N_bins = 100   #Number of bins used for the fit

N_bins_lst = np.linspace(50, 300, 51)
plt.figure()
for N_bins in N_bins_lst :
    tau, tau_err, chi2, chi2_err = Analyse(corr_t, b, dt_max, round(N_bins))
    plt.scatter(N_bins, tau, c = 'k')
    plt.errorbar(N_bins, tau, yerr = tau_err, linestyle = 'None', capsize=4, c='k')
plt.xlabel(r'$N_{bins}$')
plt.ylabel(r'$\tau$ [µs]')
#plt.text(45, 2.23, "$t_{corr}$ = 25 ns\n$b$ = 12.5 $10^4$ arbitrary unit\n$N_{bins}$ = 100")
plt.grid()
plt.title(r'$\tau$ Result For Different $N_{bins}$')
plt.savefig('N_bins')