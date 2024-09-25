import numpy as np
import BigFunctions as BF
from scipy.optimize import curve_fit
import uproot as ur
import warnings
warnings.filterwarnings('ignore')

#-------------------------------------------------Reading the data with uproot

print('########## Reading the Data ##########\n')

input_path = "../data/bigdata_CFD_20092024_0001.root"
with ur.open(input_path) as file:
    outTree = file["DataTree"]
    label = outTree["label"].array(library = "np")
    time = outTree["time"].array(library = "np")*10**-6   #[µs]
    nrj = outTree["nrj"].array(library = "np")
    pileup = outTree["pileup"].array(library = "np")

print("Duration of the data acquisition:  ", max(time)*10**(-6)/3600, "[hours]")
print("Number of events acquired:         ", len(label)/10**6, "[million] (100 %)")
print("Number of events acquired with PM2:", len(label[label == 2])/10**6, "[million] (100 %)")

print('\n######################################\n')

#-------------------------------------------------Correlation

print('########## Looking at Correlation ##########\n')

corr_t = 10*10**-3   #Correlation window
print("Correlation time window:                             ", corr_t*10**3, '[ns]')
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

print("Dead time after correlation:                         ", min(dt2), "[µs]")
print("Number of events after correlation:                  ", len(t)/10**6, "[million] (", len(t)*100/len(label), "% )")
print("Number of events acquired with PM2 after correlation:", len(l[l == 2])/10**6, "[million] (", len(l[l == 2])*100/len(label[label == 2]), "% )")

print('\n############################################\n')

#-------------------------------------------------Cut on dt

print('########## Correlation + Cut on dt ##########\n')

dt_max = 20
dt = dt2[(dt2<dt_max)]
print("Number of events:", len(dt)/10**6, "[million] (", len(dt)*100/len(label[label == 2]), "% )")
print('dt_max:        ', dt_max, '[µs]')
N_bins = 100
print("Number of bins:  ", 100)

hist = np.histogram(dt, bins = N_bins)
x = hist[1][1:]-(hist[1][1]-hist[1][0])/2
y = hist[0]
y_err = np.sqrt(y)
popt, cov = curve_fit(BF.exp2, x, y, sigma=y_err, absolute_sigma = True, p0 = np.array([20000, 2, 2, 400]))
N0 = popt[0]
lam1 = popt[1]
lam2 = popt[2]
CC = popt[3]
lam1_err = np.sqrt(cov[1,1])
lam2_err = np.sqrt(cov[2,2])
chi2 = BF.chi2_norm(y, BF.exp2(x, N0, lam1, lam2, CC), y_err, 4)[0]
chi2_err = BF.chi2_norm(y, BF.exp2(x, N0, lam1, lam2, CC), y_err, 4)[1]
tau1 = 1/lam1
tau1_err = lam1_err/lam1**2
tau2 = 1/lam2
tau2_err = lam2_err/lam2**2

print('tau1 =           ', tau1, '+/-', tau1_err, '[µs]')
print('tau2 =           ', tau2, '+/-', tau2_err, '[µs]')
print('Reduced chi2:    ', chi2, '+/-', chi2_err)

print('\n#############################################\n')

#-------------------------------------------------Cut on energy

print('########## Correlation + Cut on dt + Cut on Energies ##########\n')

dt_max = 20
nrj_max = 0.25e6
dt = dt2[(dt2<dt_max) & (nrj2[:-1]<nrj_max) & (nrj2[1:]<nrj_max)]
print("Number of events:", len(dt)/10**6, "[million] (", len(dt)*100/len(label[label == 2]), "% )")
print('dt_max:        ', dt_max, '[µs]')
print('nrj_max:       ', nrj_max, '[Arbitrary Unit]')
N_bins = 100
print("Number of bins:", 100)

hist = np.histogram(dt, bins = N_bins)
x = hist[1][1:]-(hist[1][1]-hist[1][0])/2
y = hist[0]
y_err = np.sqrt(y)
popt, cov = curve_fit(BF.exp2, x, y, sigma=y_err, absolute_sigma = True, p0 = np.array([20000, 2, 2, 400]))
N0 = popt[0]
lam1 = popt[1]
lam2 = popt[2]
CC = popt[3]
lam1_err = np.sqrt(cov[1,1])
lam2_err = np.sqrt(cov[2,2])
chi2 = BF.chi2_norm(y, BF.exp2(x, N0, lam1, lam2, CC), y_err, 4)[0]
chi2_err = BF.chi2_norm(y, BF.exp2(x, N0, lam1, lam2, CC), y_err, 4)[1]
tau1 = 1/lam1
tau1_err = lam1_err/lam1**2
tau2 = 1/lam2
tau2_err = lam2_err/lam2**2

print('tau1 =         ', tau1, '+/-', tau1_err, '[µs]')
print('tau2 =         ', tau2, '+/-', tau2_err, '[µs]')
print('Reduced chi2:  ', chi2, '+/-', chi2_err)

print('\n#####################################################\n')

print('########## Correlation + Cut on dt + Cut on Energies + Cut on nrj_µ ##########\n')

dt_max = 20
nrj_max = 0.25e6
dt = dt2[(dt2<dt_max) & (nrj2[:-1]<nrj_max) & (nrj2[1:]<50000)]
print("Number of events:", len(dt)/10**6, "[million] (", len(dt)*100/len(label[label == 2]), "% )")
print('dt_max:        ', dt_max, '[µs]')
print('nrj_max:       ', nrj_max, '[Arbitrary Unit]')
N_bins = 100
print("Number of bins:", 100)

hist = np.histogram(dt, bins = N_bins)
x = hist[1][1:]-(hist[1][1]-hist[1][0])/2
y = hist[0]
y_err = np.sqrt(y)
popt, cov = curve_fit(BF.exp2, x, y, sigma=y_err, absolute_sigma = True, p0 = np.array([20000, 2, 2, 400]))
N0 = popt[0]
lam1 = popt[1]
lam2 = popt[2]
CC = popt[3]
lam1_err = np.sqrt(cov[1,1])
lam2_err = np.sqrt(cov[2,2])
chi2 = BF.chi2_norm(y, BF.exp2(x, N0, lam1, lam2, CC), y_err, 4)[0]
chi2_err = BF.chi2_norm(y, BF.exp2(x, N0, lam1, lam2, CC), y_err, 4)[1]
tau1 = 1/lam1
tau1_err = lam1_err/lam1**2
tau2 = 1/lam2
tau2_err = lam2_err/lam2**2

print('tau1 =         ', tau1, '+/-', tau1_err, '[µs]')
print('tau2 =         ', tau2, '+/-', tau2_err, '[µs]')
print('Reduced chi2:  ', chi2, '+/-', chi2_err)

print('\n##############################################################################\n')