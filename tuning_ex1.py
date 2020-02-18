
import numpy as np
import examples as exa
from integrators import CM, RGD, NAG, CRGD

def tuning_process(method, seed, flag, n_times, steps = 200):
    '''
    In this function, we do the random parameter search for each method (CM, NAG, RGD, CRGD).
    '''
    min_f = 1e+16
    ex = exa.Example1(seed)
    mu_new = 0
    dt_new = 0
    m_new = 0
    vc_new = 0

    for i in range(n_times):
        # -------------    CM    -------------
        if flag == 1:
            dt = np.random.uniform(1e-2, 0.8)
            mu = np.random.uniform(0.8, 0.999)
            ite_x = method(ex, mu, dt, steps)
            m = 0
            vc = 0
        # -------------    NAG    -------------
        elif flag == 2:
            dt = np.random.uniform(1e-3, 0.5)
            mu = np.random.uniform(0.8, 0.999)
            ite_x = method(ex, mu, dt, steps)
            m = 0
            vc = 0
        # -------------    RGD    -------------
        elif flag == 3:
            dt = np.random.uniform(1e-3, 0.5)
            mu = np.random.uniform(0.6, 0.95)
            vc = np.random.uniform(1e+3, 1e+6)
            m = np.random.uniform(1e-4, 1e-2)
            ite_x = method(ex, mu, dt, vc, m, steps)
        # -------------    CRGD    -------------
        else:
            dt = np.random.uniform(0.1, 0.6)
            mu = np.random.uniform(0.39, 0.9)
            vc = np.random.uniform(1e+4, 1e+7)
            m = np.random.uniform(1e-3, 1e-1)
            ite_x = method(ex, [vc, m, mu], dt, steps)

        f_sim = np.apply_along_axis(ex.f, 1, ite_x)
        min_fnew = min(f_sim)
        print('minfnew', min_fnew, 'mu', mu, 'dt', dt, 'm', m, 'c', vc, 'mu', mu)

        if min_fnew < min_f:
            min_f = min_fnew
            mu_new = mu
            dt_new = dt
            m_new = m
            vc_new = vc
    return mu_new, dt_new, m_new, vc_new


def find_params(n_times, num_trials, method, flag):
    '''
    In this function, we tuned the parameters for Figure 1. We run the function
    tuning_process() for each sample of A.
    '''
    Mu = []
    Dt = []
    M = []
    Vc = []

    for i in range(num_trials):
        mu, dt, m, vc = tuning_process(method, i, flag, n_times)
        Mu.append(mu)
        Dt.append(dt)
        M.append(m)
        Vc.append(vc)
    return Mu, Dt, M, Vc


def params_Fig1(n_times, num_trials):
    '''
    In this function, we called the tuned parameters for Figure 3.
    '''
    mu_CM, dt_CM, _, _ = find_params(n_times, num_trials, CM, 1)
    mu_NAG, dt_NAG, _, _ = find_params(n_times, num_trials, NAG, 2)
    mu_RGD, dt_RGD, m_RGD, vc_RGD = find_params(n_times, num_trials, RGD, 3)
    mu_CRGD, dt_CRGD, m_CRGD, vc_CRGD = find_params(n_times, num_trials, CRGD, 4)

    np.save("mu_CM_ex1", mu_CM)
    np.save("dt_CM_ex1", dt_CM)

    np.save("mu_NAG_ex1", mu_NAG)
    np.save("dt_NAG_ex1", dt_NAG)

    np.save("mu_RGD_ex1", mu_RGD)
    np.save("dt_RGD_ex1", dt_RGD)
    np.save("m_RGD_ex1", m_RGD)
    np.save("vc_RGD_ex1", vc_RGD)

    np.save("mu_CRGD_ex1", mu_CRGD)
    np.save("dt_CRGD_ex1", dt_CRGD)
    np.save("m_CRGD_ex1", m_CRGD)
    np.save("vc_CRGD_ex1", vc_CRGD)


# -------- Tuning process for example 1 ---------
# To simulate  parameters for Example 1, uncommented the following line
# params_Fig1(n_times=100, num_trials=50)
