
import numpy as np
import examples as exa
import matplotlib.pyplot as plt
from integrators import CM, RGD, NAG, CRGD
plt.rcParams['font.size'] = 20

# Load tuned parameters for Example 1. You can generate these parameters using tuning_ex1.py

mu_CM = np.loadtxt("params/params_ex1/mu_CM_ex1.txt")
dt_CM = np.loadtxt("params/params_ex1/dt_CM_ex1.txt")

mu_NAG = np.loadtxt("params/params_ex1/mu_NAG_ex1.txt")
dt_NAG = np.loadtxt("params/params_ex1/dt_NAG_ex1.txt")

mu_RGD = np.loadtxt("params/params_ex1/mu_RGD_ex1.txt")
dt_RGD = np.loadtxt("params/params_ex1/dt_RGD_ex1.txt")
m_RGD = np.loadtxt("params/params_ex1/m_RGD_ex1.txt")
vc_RGD = np.loadtxt("params/params_ex1/vc_RGD_ex1.txt")

mu_CRGD = np.load("params/params_ex1/mu_CRGD_ex1.npy")
dt_CRGD = np.load("params/params_ex1/dt_CRGD_ex1.npy")
m_CRGD = np.load("params/params_ex1/m_CRGD_ex1.npy")
vc_CRGD = np.load("params/params_ex1/vc_CRGD_ex1.npy")


def plot_Fig1(num_trials, steps):  # Figure 1

    plt.figure(figsize=(12, 9))
    F_simu_CM = np.empty([num_trials, steps], dtype=np.float64)
    F_simu_NAG = np.empty([num_trials, steps], dtype=np.float64)
    F_simu_RGD = np.empty([num_trials, steps], dtype=np.float64)
    F_simu_CRGD = np.empty([num_trials, steps], dtype=np.float64)

    for i in range(num_trials):
        ex = exa.Example1(i)
        sol_x_CM = CM(ex, mu_CM[i], dt_CM[i], steps)
        sol_x_NAG = NAG(ex, mu_NAG[i], dt_NAG[i], steps)
        sol_x_RGD = RGD(ex, mu_RGD[i], dt_RGD[i], vc_RGD[i], m_RGD[i], steps)
        sol_x_CRGD = CRGD(ex, [vc_CRGD[i], m_CRGD[i], mu_CRGD[i]], dt_CRGD[i], steps)

        F_simu_CM[i] = np.apply_along_axis(ex.f, 1, sol_x_CM)
        F_simu_NAG[i] = np.apply_along_axis(ex.f, 1, sol_x_NAG)
        F_simu_RGD[i] = np.apply_along_axis(ex.f, 1, sol_x_RGD)
        F_simu_CRGD[i] = np.apply_along_axis(ex.f, 1, sol_x_CRGD)

        plt.plot(F_simu_CM[i], color='b',  linewidth=2)
        plt.plot(F_simu_NAG[i], color='g', linewidth=2)
        plt.plot(F_simu_RGD[i], color='r', linewidth=2)
        plt.plot(F_simu_CRGD[i], color='k', linewidth=2)

    plt.plot(F_simu_CM[-1], color='b', linewidth=3, label='CM')
    plt.plot(F_simu_NAG[-1], color='g', linewidth=3, label='NAG')
    plt.plot(F_simu_RGD[-1], color='r', linewidth=3, label='RGD')
    plt.plot(F_simu_CRGD[-1], color='k', linewidth=3, label='CRGD')

    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Function value', fontsize=30)
    plt.yscale('log')
    plt.xticks([0, 100, 200, 300])
    plt.yticks([1e+2, 1e-14, 1e-30, 1e-46, 1e-62])
    plt.xlim(-4, 304)
    plt.legend(loc=3, fontsize=21, frameon=False)

    plt.savefig("Figure1.eps")


def plot_Fig2(steps):  # Figure 2

    plt.figure(figsize=(12, 9))
    ex = exa.Example1(0)
    dt = 0.53  # the step size is changed

    sol_x_RGD = RGD(ex, mu_RGD[0], dt_RGD[0], vc_RGD[0], m_RGD[0], steps)
    sol_x_CRGD = CRGD(ex, [vc_RGD[0], m_RGD[0], mu_RGD[0]], dt_RGD[0], steps)

    sol_x_CRGD_ = CRGD(ex, [vc_RGD[0], m_RGD[0], mu_RGD[0]], dt, steps)
    sol_x_RGD_ = RGD(ex, mu_RGD[0], dt, vc_RGD[0], m_RGD[0], steps)

    F_simu_RGD = np.apply_along_axis(ex.f, 1, sol_x_RGD)
    F_simu_RGD_ = np.apply_along_axis(ex.f, 1, sol_x_RGD_)
    F_simu_CRGD= np.apply_along_axis(ex.f, 1, sol_x_CRGD)
    F_simu_CRGD_ = np.apply_along_axis(ex.f, 1, sol_x_CRGD_)

    plt.plot(F_simu_RGD, marker='o', linewidth=2, markevery=50, markersize=16, fillstyle='none', color='r', label='RGD,' + r'$\tau$=0.43')
    plt.plot(F_simu_RGD_, marker='<', linewidth=2, markevery=50, markersize=16, fillstyle='none', color='r', label='RGD,'+ r'$\tau$=0.53')
    plt.plot(F_simu_CRGD, marker='H', linewidth=2, markevery=50, markersize=16, fillstyle='none', color='k', label='CRGD,'+ r'$\tau$=0.43')
    plt.plot(F_simu_CRGD_, marker='>', linewidth=2, markevery=50, markersize=16, fillstyle='none', color='k', label='CRGD,'+ r'$\tau$=0.53')
    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Function value', fontsize=30)
    plt.yscale('log')
    plt.xticks([0, 100, 200, 300])
    plt.yticks([1e+2, 1e-14, 1e-30, 1e-46, 1e-62])
    plt.xlim(-4, 304)
    plt.legend(loc=3, fontsize=22, frameon=False)
    plt.savefig("Figure2.eps")


# -------- Figures for Example 1 ---------
# To plot Figures 1,2 , uncommented the following lines,
#
# plot_Fig1(num_trials=50, steps=300)  # Figure 1
# plot_Fig2(steps=300)  # Figure 2



