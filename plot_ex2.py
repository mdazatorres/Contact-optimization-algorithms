import numpy as np
import examples as exa
import matplotlib.pyplot as plt
from integrators import CM, RGD, NAG, CRGD
plt.rcParams['font.size'] = 20

# Load tuned parameters for Example 2. You can generate these parameters using tuning_ex2.py

mu_CM_mean = np.load("params/params_ex2/mu_CM_ex2.npy")
dt_CM_mean = np.load("params/params_ex2/dt_CM_ex2.npy")

mu_NAG_mean = np.load("params/params_ex2/mu_NAG_ex2.npy")
dt_NAG_mean = np.load("params/params_ex2/dt_NAG_ex2.npy")

mu_RGD_mean = np.load("params/params_ex2/mu_RGD_ex2.npy")
dt_RGD_mean = np.load("params/params_ex2/dt_RGD_ex2.npy")
m_RGD_mean = np.load("params/params_ex2/m_RGD_ex2.npy")
vc_RGD_mean = np.load("params/params_ex2/vc_RGD_ex2.npy")

mu_CRGD_mean = np.load("params/params_ex2/mu_CRGD_ex2.npy")
dt_CRGD_mean = np.load("params/params_ex2/dt_CRGD_ex2.npy")
m_CRGD_mean = np.load("params/params_ex2/m_CRGD_ex2.npy")
vc_CRGD_mean = np.load("params/params_ex2/vc_CRGD_ex2.npy")


def plot_Fig3(steps, num_trials):

    plt.figure(figsize=(12, 9))
    F_simu_CM = np.empty([num_trials, steps], dtype=np.float64)
    F_simu_NAG = np.empty([num_trials, steps], dtype=np.float64)
    F_simu_RGD = np.empty([num_trials, steps], dtype=np.float64)
    F_simu_CRGD = np.empty([num_trials, steps], dtype=np.float64)

    for i in range(num_trials):
        ex = exa.Example2(i)
        sol_x_CM = CM(ex, mu_CM_mean, dt_CM_mean, steps)
        sol_x_NAG = NAG(ex, mu_NAG_mean, dt_NAG_mean, steps)
        sol_x_RGD = RGD(ex, mu_RGD_mean, dt_RGD_mean, vc_RGD_mean, m_RGD_mean, steps)
        sol_x_CRGD = CRGD(ex, [vc_CRGD_mean, m_CRGD_mean, mu_CRGD_mean], dt_CRGD_mean, steps)

        f_sim_CM = np.apply_along_axis(ex.f, 1, sol_x_CM)
        f_sim_NAG = np.apply_along_axis(ex.f, 1, sol_x_NAG)
        f_sim_RGD = np.apply_along_axis(ex.f, 1, sol_x_RGD)
        f_sim_CRGD = np.apply_along_axis(ex.f, 1, sol_x_CRGD)

        F_simu_CM[i] = f_sim_CM
        F_simu_NAG[i] = f_sim_NAG
        F_simu_RGD[i] = f_sim_RGD
        F_simu_CRGD[i] = f_sim_CRGD

        plt.plot(F_simu_CM[i], color='b', linewidth=2)
        plt.plot(F_simu_NAG[i], color='g', linewidth=2)
        plt.plot(F_simu_RGD[i], color='r', linewidth=2)
        plt.plot(F_simu_CRGD[i], color='k', linewidth=2)

    plt.plot(F_simu_CM[-1], color='b', linewidth=3, label='CM')
    plt.plot(F_simu_NAG[-1], color='g', linewidth=3, label='NAG')
    plt.plot(F_simu_RGD[-1], color='r', linewidth=3,  label='RGD')
    plt.plot(F_simu_CRGD[-1], color='k', linewidth=3, label='CRGD')

    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Function value', fontsize=30)
    plt.yscale('log')
    plt.xticks([0, 100, 200, 300])
    plt.yticks([1e+2, 1e-6, 1e-14, 1e-22])
    plt.xlim(-4, 304)
    plt.legend(loc=3, fontsize=21, frameon=False)
    plt.savefig("Figure3.eps")


# -------- Figures for Example 2 ---------
# To plot Figure 3 , uncommented the following line
# plot_Fig3(steps=300, num_trials=50)  # Figure 3






