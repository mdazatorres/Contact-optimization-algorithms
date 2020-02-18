import numpy as np
import examples as exa
import matplotlib.pyplot as plt
from integrators import CM, RGD, NAG, CRGD
import pylab
plt.rcParams['font.size'] = 20


def plot_Fig5a():
    '''Contour plot for Rosenbrock function
    '''
    x0 = np.random.uniform(-2.048, 2.048, 100)
    ex = exa.Example4(x0, seed=1)
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    plt.figure(figsize=(12, 9))

    x_ = np.linspace(-1.3, 1.3, 300)
    y_ = np.linspace(-1.3, 1.3, 300)
    X, Y = np.meshgrid(x_, y_)

    cs = pylab.contourf(X, Y, ex.f2D(X, Y), levels=23, cmap='PuBu')
    plt.contour(X, Y, ex.f2D(X, Y), levels=23, linewidths=1.0, colors='k')
    cbar = pylab.colorbar(cs)
    cbar_ticks = [0, 240, 480, 720, 960]
    cbar.ax.set_autoscale_on(True)
    cbar.set_ticks(cbar_ticks)
    plt.xlabel('$X_1$', fontsize=30)
    plt.ylabel('$X_2$', fontsize=30)
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.savefig("Figure5a.eps")


def plot_Fig5b(steps):

    # Load tuned parameters for Figure 5b. You can generate these parameters using tuning_ex4.py
    mu_CM = np.load("params/params_ex4/mu_CM_ex4b.npy")
    dt_CM = np.load("params/params_ex4/dt_CM_ex4b.npy")

    mu_NAG = np.load("params/params_ex4/mu_NAG_ex4b.npy")
    dt_NAG = np.load("params/params_ex4/dt_NAG_ex4b.npy")

    mu_RGD = np.load("params/params_ex4/mu_RGD_ex4b.npy")
    dt_RGD = np.load("params/params_ex4/dt_RGD_ex4b.npy")
    m_RGD = np.load("params/params_ex4/m_RGD_ex4b.npy")
    vc_RGD = np.load("params/params_ex4/vc_RGD_ex4b.npy")

    mu_CRGD = np.load("params/params_ex4/mu_CRGD_ex4b.npy")
    dt_CRGD = np.load("params/params_ex4/dt_CRGD_ex4b.npy")
    m_CRGD = np.load("params/params_ex4/m_CRGD_ex4b.npy")
    vc_CRGD = np.load("params/params_ex4/vc_CRGD_ex4b.npy")

    n = 100  # Dimension for the Rosenbrock function
    x0 = np.zeros(n)
    x0[::2] = -1.2
    x0[1::2] = 1
    ex = exa.Example4(x0, seed=1)

    sol_x_CM = CM(ex, mu_CM, dt_CM, steps)
    sol_x_NAG = NAG(ex, mu_NAG, dt_NAG, steps)
    sol_x_RGD = RGD(ex, mu_RGD, dt_RGD, vc_RGD, m_RGD, steps)
    sol_x_CRGD = CRGD(ex, [vc_CRGD, m_CRGD, mu_CRGD], dt_CRGD, steps)

    f_sim_CM = np.apply_along_axis(ex.f, 1, sol_x_CM)
    f_sim_NAG = np.apply_along_axis(ex.f, 1, sol_x_NAG)
    f_sim_RGD = np.apply_along_axis(ex.f, 1, sol_x_RGD)
    f_sim_CRGD = np.apply_along_axis(ex.f, 1, sol_x_CRGD)

    plt.figure(figsize=(12, 9))
    plt.plot(f_sim_CM, linewidth=2,  color='b', marker='D', markevery=80, label='CM')
    plt.plot(f_sim_NAG, linewidth=2, color='g', marker='s', markevery=80, label='NAG')
    plt.plot(f_sim_RGD, linewidth=2, color='r', marker='o', markevery=80, label='RGD')
    plt.plot(f_sim_CRGD, linewidth=2, color='k', marker='H', markevery=80, label='CRGD')
    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Function value', fontsize=30)
    plt.yscale('log')
    plt.xticks([0, 500, 1000, 1500])  # work on current fig
    plt.yticks([1e+4, 1e-6, 1e-16])  # work on current fig
    plt.xlim([-10, 1510])
    plt.legend(loc=3, fontsize=21, frameon=False)
    plt.savefig("Figure4b.eps")


def plot_Fig5c(steps):

    # Load tuned parameters for Figure 5c. You can generate these parameters using tuning_ex4.py
    mu_CM = np.load("params/params_ex4/mu_CM_ex4c.npy")
    dt_CM = np.load("params/params_ex4/dt_CM_ex4c.npy")

    mu_NAG = np.load("params/params_ex4/mu_NAG_ex4c.npy")
    dt_NAG = np.load("params/params_ex4/dt_NAG_ex4c.npy")

    mu_RGD = np.load("params/params_ex4/mu_RGD_ex4c.npy")
    dt_RGD = np.load("params/params_ex4/dt_RGD_ex4c.npy")
    m_RGD = np.load("params/params_ex4/m_RGD_ex4c.npy")
    vc_RGD = np.load("params/params_ex4/vc_RGD_ex4c.npy")

    mu_CRGD = np.load("params/params_ex4/mu_CRGD_ex4c.npy")
    dt_CRGD = np.load("params/params_ex4/dt_CRGD_ex4c.npy")
    m_CRGD = np.load("params/params_ex4/m_CRGD_ex4c.npy")
    vc_CRGD = np.load("params/params_ex4/vc_CRGD_ex4c.npy")

    n = 100  # Dimension for the Rosenbrock function
    x0 = np.zeros(n)
    x0[::2] = -5
    x0[1::2] = 5
    ex = exa.Example4(x0, seed=1)

    sol_x_CM = CM(ex, mu_CM, dt_CM, steps)
    sol_x_NAG = NAG(ex, mu_NAG, dt_NAG, steps)
    sol_x_RGD = RGD(ex, mu_RGD, dt_RGD, vc_RGD, m_RGD, steps)
    sol_x_CRGD = CRGD(ex, [vc_CRGD, m_CRGD, mu_CRGD], dt_CRGD, steps)

    f_sim_CM = np.apply_along_axis(ex.f, 1, sol_x_CM)
    f_sim_NAG = np.apply_along_axis(ex.f, 1, sol_x_NAG)
    f_sim_RGD = np.apply_along_axis(ex.f, 1, sol_x_RGD)
    f_sim_CRGD = np.apply_along_axis(ex.f, 1, sol_x_CRGD)

    plt.figure(figsize=(12, 9))
    plt.plot(f_sim_CM, linewidth=2,  color='b', marker='D', markevery=80, label='CM')
    plt.plot(f_sim_NAG, linewidth=2, color='g', marker='s', markevery=80, label='NAG')
    plt.plot(f_sim_RGD, linewidth=2, color='r', marker='o', markevery=80, label='RGD')
    plt.plot(f_sim_CRGD, linewidth=2, color='k', marker='H', markevery=80, label='CRGD')
    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Function value', fontsize=30)
    plt.yscale('log')
    plt.xticks([0, 1000, 2000, 3000, 4000])
    plt.yticks([1e+6, 1e-4, 1e-14, 1e-24])
    plt.xlim([-20, 4020])
    plt.legend(loc=3, fontsize=21, frameon=False)
    plt.savefig("Figure4c.eps")


def plot_Fig5d(num_samples, steps):  # Figure 5d

    # Load tuned parameters for Figure 5d. You can generate these parameters using tuning_ex4.py
    mu_CM_mean = np.load("params/params_ex4/mu_CM_ex4d.npy")
    dt_CM_mean = np.load("params/params_ex4/dt_CM_ex4d.npy")

    mu_NAG_mean = np.load("params/params_ex4/mu_NAG_ex4d.npy")
    dt_NAG_mean = np.load("params/params_ex4/dt_NAG_ex4d.npy")

    mu_RGD_mean = np.load("params/params_ex4/mu_RGD_ex4d.npy")
    dt_RGD_mean = np.load("params/params_ex4/dt_RGD_ex4d.npy")
    m_RGD_mean = np.load("params/params_ex4/m_RGD_ex4d.npy")
    vc_RGD_mean = np.load("params/params_ex4/vc_RGD_ex4d.npy")

    mu_CRGD_mean = np.load("params/params_ex4/mu_CRGD_ex4d.npy")
    dt_CRGD_mean = np.load("params/params_ex4/dt_CRGD_ex4d.npy")
    m_CRGD_mean = np.load("params/params_ex4/m_CRGD_ex4d.npy")
    vc_CRGD_mean = np.load("params/params_ex4/vc_CRGD_ex4d.npy")

    F_simu_CM = 0
    F_simu_NAG = 0
    F_simu_RGD = 0
    F_simu_CRGD = 0

    plt.figure(figsize=(12, 9))
    n = 100  # Dimension for the Rosenbrock function
    for i in range(num_samples):

        np.random.seed(i)
        x0 = np.random.uniform(-2.048, 2.048, n)
        ex = exa.Example4(x0, seed=i)

        sol_x_CM = CM(ex, mu_CM_mean, dt_CM_mean, steps)
        sol_x_NAG = NAG(ex, mu_NAG_mean, dt_NAG_mean, steps)
        sol_x_RGD = RGD(ex, mu_RGD_mean, dt_RGD_mean, vc_RGD_mean, m_RGD_mean, steps)
        sol_x_CRGD = CRGD(ex, [vc_CRGD_mean, m_CRGD_mean, mu_CRGD_mean], dt_CRGD_mean, steps)

        if any(np.apply_along_axis(ex.f, 1, sol_x_CM) < 1e-1):
            F_simu_CM = np.apply_along_axis(ex.f, 1, sol_x_CM)
            plt.plot(F_simu_CM, color='b', alpha=0.5)

        if any(np.apply_along_axis(ex.f, 1, sol_x_NAG) < 1e-2):
            F_simu_NAG = np.apply_along_axis(ex.f, 1, sol_x_NAG)
            plt.plot(F_simu_NAG, color='g', alpha=0.5)

        if any(np.apply_along_axis(ex.f, 1, sol_x_RGD) < 1e-1):
            F_simu_RGD = np.apply_along_axis(ex.f, 1, sol_x_RGD)
            plt.plot(F_simu_RGD, color='r', alpha=0.5)

        if any(np.apply_along_axis(ex.f, 1, sol_x_CRGD) < 1e-1):
            F_simu_CRGD = np.apply_along_axis(ex.f, 1, sol_x_CRGD)
            plt.plot(F_simu_CRGD, color='k', alpha=0.5)

    plt.plot(F_simu_CM, color='b', label='CM')
    plt.plot(F_simu_NAG, color='g', label='NAG')
    plt.plot(F_simu_RGD, color='r', label='RGD')
    plt.plot(F_simu_CRGD, color='k', label='CRGD')
    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Function value', fontsize=30)
    plt.yscale('log')
    plt.legend(loc=3, fontsize=21, frameon=False)
    plt.savefig("Figure5d.eps")


# -------- Figures for Example 4 ---------
# To plot Figure 5a-d , uncommented the following lines

# plot_Fig5a()
# plot_Fig5b(steps = 1500)
# plot_Fig5c(steps = 4000)
# plot_Fig5d(num_samples=200, steps=3000)


