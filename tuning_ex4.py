
import numpy as np
import examples as exa
from integrators import CM, RGD, NAG, CRGD
from matplotlib import rcParams
rcParams['font.size'] = 20

def tuning_process(method, flag, n_times, steps, x0, seed):
    '''
    In this function, we do the random parameter search for each method (CM, NAG, RGD, CRGD).
    '''
    min_f = 1e+16
    mu_new = 0
    dt_new = 0
    m_new = 0
    vc_new = 0

    ex = exa.Example4(x0, seed)
    for i in range(n_times):
        # -------------    CM    -------------
        if flag == 1:

            #  parameters figures 4b-c
            dt = np.random.uniform(1e-5, 1e-3)
            mu = np.random.uniform(0.9, 0.98)

            #  parameters figures 4d
            # dt = np.random.uniform(2 * 1e-4, 4 * 1e-4)
            # mu = np.random.uniform(0.94, 0.98)
            ite_x = method(ex, mu, dt, steps)
            m = 0
            vc = 0
        # -------------    NAG    -------------
        elif flag == 2:

            #  parameters figures 4b-c
            dt = np.random.uniform(1e-5, 1e-3)
            mu = np.random.uniform(0.9, 0.98)

            #  parameters figure 4d (uncommented the following lines for parameters of figure 4d)
            # dt = np.random.uniform(2 * 1e-4, 4 * 1e-4)
            # mu = np.random.uniform(0.94, 0.98)

            ite_x = method(ex, mu, dt, steps)
            m = 0
            vc = 0
        # -------------    RGD    -------------
        elif flag == 3:

            #  parameters figures 4b-c
            dt = np.random.uniform(1e-5, 8*1e-3)
            mu = np.random.uniform(0.9, 0.98)
            vc = np.random.uniform(1e+3, 1e+5)
            m = np.random.uniform(1e-6, 1e-4)

            #  parameters figures 4d (uncommented the following lines for parameters of figure 4d)
            # dt = np.random.uniform(1e-5, 1e-4)
            # mu = np.random.uniform(0.93, 0.97)
            # vc = np.random.uniform(1e+4, 9*1e+4)
            # m = np.random.uniform(4*1e-7, 1e-6)

            ite_x = method(ex, mu, dt, vc, m, steps)
        # -------------    CRGD    -------------
        else:
            #  parameters figures 4b-d
            dt = np.random.uniform(1e-5, 8*1e-3)
            mu = np.random.uniform(0.9, 0.98)
            vc = np.random.uniform(1e+3, 1e+5)
            m = np.random.uniform(1e-5, 1e-2)

            ite_x = method(ex, [vc, m, mu], dt, steps)

        f_sim = np.apply_along_axis(ex.f, 1, ite_x)
        min_fnew = min(f_sim)
        print('minfnew', min_fnew, 'mu', mu, 'dt', dt,  'm', m, 'vc', vc)

        if min_fnew < min_f:
            min_f = min_fnew
            mu_new = mu
            dt_new = dt
            m_new = m
            vc_new = vc

    return mu_new, dt_new, m_new, vc_new


def params_Fig5b(steps_trial, n_times):
    '''
    In this function, we tuned the parameters
    for Figure 5b. Before running this function uncomment
    the respective  range search for the parameters of Figure 5b in tuning_process()
    '''
    n = 100  # dimension for Rosenbrock function
    x0 = np.zeros(n)
    x0[::2] = -1.2
    x0[1::2] = 1

    mu_CM, dt_CM, _, _ = tuning_process(CM, 1,  n_times, steps_trial, x0, seed=1)
    mu_NAG, dt_NAG, _, _ = tuning_process(NAG, 2, n_times, steps_trial, x0, seed=1)
    mu_RGD, dt_RGD, m_RGD, vc_RGD = tuning_process(RGD, 3,  n_times, steps_trial, x0, seed=1)
    mu_CRGD, dt_CRGD, m_CRGD, vc_CRGD = tuning_process(CRGD, 5, n_times, steps_trial, x0, seed=1)

    np.save("mu_CM_ex4b", mu_CM)
    np.save("dt_CM_ex4b", dt_CM)

    np.save("mu_NAG_ex4b", mu_NAG)
    np.save("dt_NAG_ex4b", dt_NAG)

    np.save("mu_RGD_ex4b", mu_RGD)
    np.save("dt_RGD_ex4b", dt_RGD)
    np.save("m_RGD_ex4b", m_RGD)
    np.save("vc_RGD_ex4b", vc_RGD)

    np.save("mu_CRGD_ex4b", mu_CRGD)
    np.save("dt_CRGD_ex4b", dt_CRGD)
    np.save("m_CRGD_ex4b", m_CRGD)
    np.save("vc_CRGD_ex4b", vc_CRGD)


def params_Fig5c(steps_trial, n_times):
    '''
    In this function, we tuned the parameters
    for Figure 5c. Before running this function uncomment
    the respective  range search for the parameters of Figure 5c in tuning_process()
    '''
    n = 100 # dimension for Rosenbrock function
    x0 = np.zeros(n)
    x0[::2] = -5
    x0[1::2] = 5

    mu_CM, dt_CM, _, _ = tuning_process(CM, 1, n_times, steps_trial, x0, seed=1)
    mu_NAG, dt_NAG, _, _ = tuning_process(NAG, 2, n_times, steps_trial, x0, seed=1)
    mu_RGD, dt_RGD, m_RGD, vc_RGD = tuning_process(RGD, 3, n_times, steps_trial, x0, seed=1)
    mu_CRGD, dt_CRGD, m_CRGD, vc_CRGD = tuning_process(CRGD, 4, n_times, steps_trial, x0, seed=1)

    np.save("mu_CM_ex4c", mu_CM)
    np.save("dt_CM_ex4c", dt_CM)

    np.save("mu_NAG_ex4c", mu_NAG)
    np.save("dt_NAG_ex4c", dt_NAG)

    np.save("mu_RGD_ex4c", mu_RGD)
    np.save("dt_RGD_ex4c", dt_RGD)

    np.save("m_RGD_ex4c", m_RGD)
    np.save("vc_RGD_ex4c", vc_RGD)

    np.save("mu_CRGD_ex4c", mu_CRGD)
    np.save("dt_CRGD_ex4c", dt_CRGD)
    np.save("m_CRGD_ex4c", m_CRGD)
    np.save("vc_CRGD_ex4c", vc_CRGD)


def find_params(n_times, num_trials, steps_trial, method, flag):
    '''
    In this function, we tuned the parameters for Figure 5d. We run the function
    tuning_process() for each initial condition x0.
    '''
    Mu = []
    Dt = []
    M = []
    Vc = []
    n = 100 # dimension
    for i in range(num_trials):
        np.random.seed(i)
        x0 = np.random.uniform(-2.048, 2.048, n)  # 4d
        mu, dt, m, vc = tuning_process(method, flag, n_times, steps_trial, x0, seed=i)
        Mu.append(mu)
        Dt.append(dt)
        M.append(m)
        Vc.append(vc)
    return Mu, Dt, M, Vc


def params_Fig5d(n_times, num_trials, steps_trial):
    '''
    In this function, we called the tuned parameters for Figure 5d.
    '''
    mu_CM, dt_CM, _, _ = find_params(n_times, num_trials, steps_trial, CM, 1)
    mu_NAG, dt_NAG, _, _ = find_params(n_times, num_trials, steps_trial, NAG, 2)
    mu_RGD, dt_RGD, m, vc = find_params(n_times, num_trials, steps_trial, RGD, 3)

    mu_CM_mean, dt_CM_mean = np.mean(mu_CM), np.mean(dt_CM)
    mu_NAG_mean, dt_NAG_mean = np.mean(mu_NAG), np.mean(dt_NAG)
    mu_RGD_mean, dt_RGD_mean, m_mean, vc_mean = np.mean(mu_RGD), np.mean(dt_RGD), np.mean(m), np.mean(vc)
    mu_CRGD, dt_CRGD, m_CRGD, vc_CRGD = find_params(n_times, num_trials, steps_trial, CRGD, 6)
    mu_CRGD_mean, dt_CRGD_mean, m_CRGD_mean, vc_CRGD_mean = np.mean(mu_CRGD), np.mean(dt_CRGD), np.mean(m_CRGD), np.mean(vc_CRGD)

    np.save("mu_CM_ex4d", mu_CM_mean)
    np.save("dt_CM_ex4d", dt_CM_mean)

    np.save("mu_NAG_ex4d", mu_NAG_mean)
    np.save("dt_NAG_ex4d", dt_NAG_mean)

    np.save("mu_RGD_ex4d", mu_RGD_mean)
    np.save("dt_RGD_ex4d", dt_RGD_mean)

    np.save("m_RGD_ex4d", m_mean)
    np.save("vc_RGD_ex4d", vc_mean)

    np.save("mu_CRGD_ex4d", mu_CRGD_mean)
    np.save("dt_CRGD_ex4d", dt_CRGD_mean)

    np.save("m_CRGD_ex4d1", m_CRGD_mean)
    np.save("vc_CRGD_ex4d1", vc_CRGD_mean)


# -------- Tuning process for Example 4 ---------
# To simulate  parameters for Example 4, uncommented the following lines

# params_Fig5b(steps_trial = 1200, n_times=500)
# params_Fig5c(steps_trial= 1200, n_times=500)
# params_Fig5d(n_times=200, num_trials=20, steps_trial=1200)


