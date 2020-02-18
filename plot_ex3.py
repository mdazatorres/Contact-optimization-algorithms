import numpy as np
import examples as exa
import matplotlib.pyplot as plt
from integrators import CM, RGD, NAG, CRGD
import pylab

plt.rcParams['font.size'] = 20


def plot_Fig4b(steps):

    # Load tuned parameters for Figure 4b. You can generate these parameters using tuning_ex3.py

    mu_CM = np.load("params/params_ex3/mu_CM_ex3b.npy")
    dt_CM = np.load("params/params_ex3/dt_CM_ex3b.npy")

    mu_NAG = np.load("params/params_ex3/mu_NAG_ex3b.npy")
    dt_NAG = np.load("params/params_ex3/dt_NAG_ex3b.npy")

    mu_RGD = np.load("params/params_ex3/mu_RGD_ex3b.npy")
    dt_RGD = np.load("params/params_ex3/dt_RGD_ex3b.npy")
    m_RGD = np.load("params/params_ex3/m_RGD_ex3b.npy")
    vc_RGD = np.load("params/params_ex3/vc_RGD_ex3b.npy")

    mu_CRGD = np.load("params/params_ex3/mu_CRGD_ex3b.npy")
    dt_CRGD = np.load("params/params_ex3/dt_CRGD_ex3b.npy")
    m_CRGD = np.load("params/params_ex3/m_CRGD_ex3b.npy")
    vc_CRGD = np.load("params/params_ex3/vc_CRGD_ex3b.npy")

    x0 = np.array([5, 5])  # Initial condition
    ex = exa.Example3(x0, seed=1)

    sol_x_CM = CM(ex, mu_CM, dt_CM, steps)
    sol_x_NAG = NAG(ex, mu_NAG, dt_NAG, steps)
    sol_x_RGD = RGD(ex, mu_RGD, dt_RGD, vc_RGD, m_RGD, steps)
    sol_x_CRGD = CRGD(ex, [vc_CRGD, m_CRGD, mu_CRGD], dt_CRGD, steps)

    f_sim_CM = np.apply_along_axis(ex.f, 1, sol_x_CM)
    f_sim_NAG = np.apply_along_axis(ex.f, 1, sol_x_NAG)
    f_sim_RGD = np.apply_along_axis(ex.f, 1, sol_x_RGD)
    f_sim_CRGD = np.apply_along_axis(ex.f, 1, sol_x_CRGD)

    plt.figure(figsize=(12, 9))
    plt.plot(f_sim_CM,  color='b', linewidth=2, marker='D', markersize=16, fillstyle='none', markevery=50, label='CM')
    plt.plot(f_sim_NAG, color='g', linewidth=2, marker='s',markersize=16, fillstyle='none', markevery=50, label='NAG')
    plt.plot(f_sim_RGD, color='r', linewidth=2, marker='o',markersize=16, fillstyle='none', markevery=50, label='RGD')
    plt.plot(f_sim_CRGD, color='k', linewidth=2, marker='H',markersize=16, fillstyle='none', markevery=50, label='CRGD')

    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Function value', fontsize=30)
    plt.yscale('log')
    plt.xticks([0, 100, 200, 300])
    plt.yticks([1e-1, 1e-39, 1e-77, 1e-115, 1e-153])
    plt.xlim([-6, 304])
    plt.legend(loc=3, fontsize=21, frameon=False)
    plt.savefig("Figure4b.eps")

    return f_sim_CM, f_sim_NAG, f_sim_RGD, f_sim_CRGD


def plot_Fig4c(steps):

    # Load tuned parameters for Figure 4c. You can generate these parameters using tuning_ex3.py
    mu_CM = np.load("params/params_ex3/mu_CM_ex3c.npy")
    dt_CM = np.load("params/params_ex3/dt_CM_ex3c.npy")

    mu_NAG = np.load("params/params_ex3/mu_NAG_ex3c.npy")
    dt_NAG = np.load("params/params_ex3/dt_NAG_ex3c.npy")

    mu_RGD = np.load("params/params_ex3/mu_RGD_ex3c.npy")
    dt_RGD = np.load("params/params_ex3/dt_RGD_ex3c.npy")
    m_RGD = np.load("params/params_ex3/m_RGD_ex3c.npy")
    vc_RGD = np.load("params/params_ex3/vc_RGD_ex3c.npy")

    mu_CRGD = np.load("params/params_ex3/mu_CRGD_ex3c.npy")
    dt_CRGD = np.load("params/params_ex3/dt_CRGD_ex3c.npy")
    m_CRGD = np.load("params/params_ex3/m_CRGD_ex3c.npy")
    vc_CRGD = np.load("params/params_ex3/vc_CRGD_ex3c.npy")

    x0 = np.array([1.8, -0.9])  # Initial condition
    ex = exa.Example3(x0, seed=1)

    sol_x_CM = CM(ex, mu_CM, dt_CM, steps)
    sol_x_NAG = NAG(ex, mu_NAG, dt_NAG, steps)
    sol_x_RGD = RGD(ex, mu_RGD, dt_RGD, vc_RGD, m_RGD, steps)
    sol_x_CRGD = CRGD(ex, [vc_CRGD, m_CRGD, mu_CRGD], dt_CRGD, steps)

    f_sim_CM = np.apply_along_axis(ex.f, 1, sol_x_CM)
    f_sim_NAG = np.apply_along_axis(ex.f, 1, sol_x_NAG)
    f_sim_RGD = np.apply_along_axis(ex.f, 1, sol_x_RGD)
    f_sim_CRGD = np.apply_along_axis(ex.f, 1, sol_x_CRGD)

    plt.figure(figsize=(12, 9))
    plt.plot(f_sim_CM,  color='b', linewidth=2, marker='D', markersize=16, fillstyle='none', markevery=10, label='CM')
    plt.plot(f_sim_NAG, color='g', linewidth=2, marker='s',markersize=16, fillstyle='none', markevery=10, label='NAG')
    plt.plot(f_sim_RGD, color='r', linewidth=2, marker='o',markersize=16, fillstyle='none', markevery=10, label='RGD')
    plt.plot(f_sim_CRGD, color='k', linewidth=2, marker='H',markersize=16, fillstyle='none', markevery=10, label='CRGD')

    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Function value', fontsize=30)
    plt.yscale('log')
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([1e-0, 1e-8, 1e-16, 1e-24, 1e-32])
    plt.xlim([-2, 50.5])
    plt.legend(loc=3, fontsize=21, frameon=False)
    plt.savefig("Figure4c.eps")

    return f_sim_CM, f_sim_NAG, f_sim_RGD, f_sim_CRGD


def plot_Fig4d(num_samples, steps):

    # Load tuned parameters for Figure 4d. You can generate these parameters using tuning_ex3.py

    mu_CM_mean = np.load("params/params_ex3/mu_CM_ex3d.npy")
    dt_CM_mean = np.load("params/params_ex3/dt_CM_ex3d.npy")

    mu_NAG_mean = np.load("params/params_ex3/mu_NAG_ex3d.npy")
    dt_NAG_mean = np.load("params/params_ex3/dt_NAG_ex3d.npy")

    mu_RGD_mean = np.load("params/params_ex3/mu_RGD_ex3d.npy")
    dt_RGD_mean = np.load("params/params_ex3/dt_RGD_ex3d.npy")
    m_RGD_mean = np.load("params/params_ex3/m_RGD_ex3d.npy")
    vc_RGD_mean = np.load("params/params_ex3/vc_RGD_ex3d.npy")

    mu_CRGD_mean = np.load("params/params_ex3/mu_CRGD_ex3d.npy")
    dt_CRGD_mean = np.load("params/params_ex3/dt_CRGD_ex3d.npy")
    m_CRGD_mean = np.load("params/params_ex3/m_CRGD_ex3d.npy")
    vc_CRGD_mean = np.load("params/params_ex3/vc_CRGD_ex3d.npy")

    count_CM = 1
    count_NAG = 1
    count_RGD = 1
    count_CRGD = 1

    f_simu_CM = 0
    f_simu_NAG = 0
    f_simu_RGD = 0
    f_simu_CRGD = 0

    plt.figure(figsize=(12, 9))
    for i in range(num_samples):
        np.random.seed(i)
        x0 = np.random.uniform(-5, 5, 2)
        ex = exa.Example3(x0, seed=i)
        sol_x_CM = CM(ex, mu_CM_mean, dt_CM_mean, steps)
        sol_x_NAG = NAG(ex, mu_NAG_mean, dt_NAG_mean, steps)
        sol_x_RGD = RGD(ex, mu_RGD_mean, dt_RGD_mean, vc_RGD_mean, m_RGD_mean, steps)
        sol_x_CRGD = CRGD(ex, [vc_CRGD_mean, m_CRGD_mean, mu_CRGD_mean], dt_CRGD_mean, steps)

        if all(np.apply_along_axis(ex.f, 1, sol_x_CM) < 10) and any(np.apply_along_axis(ex.f, 1, sol_x_CM) < 10e-5):
            count_CM += 1
            f_simu_CM = np.apply_along_axis(ex.f, 1, sol_x_CM)
            plt.plot(f_simu_CM, color='b', linewidth=2)

        if all(np.apply_along_axis(ex.f, 1, sol_x_NAG) < 10) and any(np.apply_along_axis(ex.f, 1, sol_x_NAG) < 10e-5):
            count_NAG += 1
            f_simu_NAG = np.apply_along_axis(ex.f, 1, sol_x_NAG)
            plt.plot(f_simu_NAG, color='g',  linewidth=2)

        if all(np.apply_along_axis(ex.f, 1, sol_x_RGD) < 10) and any(np.apply_along_axis(ex.f, 1, sol_x_RGD) < 10e-5):
            count_RGD += 1
            f_simu_RGD = np.apply_along_axis(ex.f, 1, sol_x_RGD)
            plt.plot(f_simu_RGD, color='r',  linewidth=2)

        if all(np.apply_along_axis(ex.f, 1, sol_x_CRGD) < 10) and any(np.apply_along_axis(ex.f, 1, sol_x_CRGD) < 10e-5):
            count_CRGD += 1
            f_simu_CRGD = np.apply_along_axis(ex.f, 1, sol_x_CRGD)
            plt.plot(f_simu_CRGD, color='k',  linewidth=2)

    plt.plot(f_simu_CM, color='b', linewidth=2, marker='D', markersize=16, fillstyle='none', markevery=50, label='CM')
    plt.plot(f_simu_NAG, color='g', linewidth=2, marker='s', markersize=16, fillstyle='none', markevery=50, label='NAG')
    plt.plot(f_simu_RGD, color='r', linewidth=2, marker='o', markersize=16, fillstyle='none', markevery= 50, label='RGD')
    plt.plot(f_simu_CRGD, color='k', linewidth=2, marker='H', markersize=16, fillstyle='none', markevery= 50, label='CRGD')

    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Function value', fontsize=30)
    plt.yscale('log')
    plt.xticks([0, 200, 400, 600])
    plt.yticks([1e-16, 1e-86, 1e-156, 1e-226, 1e-296])
    plt.xlim(-10, 610)
    plt.legend(loc=3, fontsize=21, frameon=False)
    plt.savefig("Figure4d.eps")
    return count_CM, count_NAG, count_RGD, count_CRGD


def plot_Fig4a():

    '''Contour plot for Camelback function
    '''
    x0 = np.random.uniform(-5, 5, 2)
    ex = exa.Example3(x0, seed=1)
    plt.rcParams['contour.negative_linestyle'] = 'solid'

    x_ = np.linspace(-2.0, 2.0, 300)
    y_ = np.linspace(-2.0, 2.0, 300)
    X, Y = np.meshgrid(x_, y_)
    plt.figure(figsize=(12, 9))
    cs = pylab.contourf(X, Y, ex.f2D(X, Y), levels=23, cmap='PuBu')
    plt.contour(X, Y, ex.f2D(X, Y), levels=23, linewidths=1.0, colors='k')
    cbar = pylab.colorbar(cs)
    cbar_ticks = [0, 3, 6, 9]
    cbar.ax.set_autoscale_on(True)
    cbar.set_ticks(cbar_ticks)

    plt.xlabel('$X_1$', fontsize=30)
    plt.ylabel('$X_2$', fontsize=30)
    plt.xticks([-2, -1, 0, 1, 2])
    plt.yticks([-2, -1, 0, 1, 2])
    plt.savefig("Figure4a.eps")


def rate_conv(f_sim_CRGD, f_sim_RGD, num_ite):
    '''
    This function compute numerically estimating the convergence rates in the iteration
     n = num ite for CRGD and RGD
    '''
    m_RGD = np.log10(f_sim_RGD[num_ite]) / np.log10(num_ite)
    m_CRGD = np.log10(f_sim_CRGD[num_ite]) / np.log10(num_ite)
    return m_RGD, m_CRGD


# -------- Figures for Example 3 ---------
# To plot Figure 4a-d , uncommented the following lines
# plot_Fig4a()
# plot_Fig4b(steps=300)
# plot_Fig4c(steps=50)
# plot_Fig4d(num_samples = 500, steps = 600)


