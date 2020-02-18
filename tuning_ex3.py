
import numpy as np
import examples as exa
from integrators import CM, RGD, NAG, CRGD


def tuning_process(method, flag, n_times, steps, x0, seed):
    '''
    In this function, we do the random parameter search for each method (CM, NAG, RGD, CRGD).
    '''
    min_f = 1e+16
    mu_new = 0
    dt_new = 0
    m_new = 0
    vc_new = 0
    ex = exa.Example3(x0, seed)

    for i in range(n_times):
        # -------------    CM    -------------
        if flag == 1:
            dt = np.random.uniform(1e-5, 1e-3)
            mu = np.random.uniform(0.8, 0.999)
            ite_x = method(ex, mu, dt, steps)
            m = 0
            vc = 0
        # -------------    NAG    -------------
        elif flag == 2:
            dt = np.random.uniform(1e-5, 1e-3)
            mu = np.random.uniform(0.8, 0.999)
            ite_x = method(ex, mu, dt, steps)
            m = 0
            vc = 0
        # -------------    RGD    -------------
        elif flag == 3:
            dt = np.random.uniform(1e-5, 8*1e-3)
            mu = np.random.uniform(0.3, 0.8)
            vc = np.random.uniform(1e+3, 1e+5)
            m = np.random.uniform(1e-6, 1e-4)
            ite_x = method(ex, mu, dt, vc, m, steps)
        # -------------    CRGD    -------------
        else:
            dt = np.random.uniform(1e-5, 8 * 1e-3)
            mu = np.random.uniform(0.15, 0.65)
            vc = np.random.uniform(1e+3, 1e+5)
            m = np.random.uniform(1e-6, 1e-4)
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


def params_Fig4b(steps, n_times):
    '''
    In this function, we tuned the parameters
    for Figure 4b.
    '''
    x0 = np.array([5, 5])  # Initial condition
    mu_CM, dt_CM, _, _ = tuning_process(CM, 1,  n_times, steps, x0, seed=1)
    mu_NAG, dt_NAG, _, _ = tuning_process(NAG, 2, n_times, steps, x0, seed=1)
    mu_RGD, dt_RGD, m_RGD, vc_RGD = tuning_process(RGD, 3,  n_times, steps, x0, seed=1)
    mu_CRGD, dt_CRGD, m_CRGD, vc_CRGD = tuning_process(CRGD, 4, n_times, steps, x0, seed=1)

    np.save("mu_CM_ex3b", mu_CM)
    np.save("dt_CM_ex3b", dt_CM)

    np.save("mu_NAG_ex3b", mu_NAG)
    np.save("dt_NAG_ex3b", dt_NAG)

    np.save("mu_RGD_ex3b", mu_RGD)
    np.save("dt_RGD_ex3b", dt_RGD)
    np.save("m_RGD_ex3b", m_RGD)
    np.save("vc_RGD_ex3b", vc_RGD)

    np.save("mu_CRGD_ex3b", mu_CRGD)
    np.save("dt_CRGD_ex3b", dt_CRGD)
    np.save("m_CRGD_ex3b", m_CRGD)
    np.save("vc_CRGD_ex3b", vc_CRGD)


def params_Fig4c(steps, n_times):
    '''
    In this function, we tuned the parameters
    for Figure 4c.
    '''
    x0 = np.array([1.8, -0.9])  # Initial condition
    mu_CM, dt_CM, _, _ = tuning_process(CM, 1,  n_times, steps, x0, seed=1)
    mu_NAG, dt_NAG, _, _ = tuning_process(NAG, 2, n_times, steps, x0, seed=1)
    mu_RGD, dt_RGD, m_RGD, vc_RGD = tuning_process(RGD, 3,  n_times, steps, x0, seed=1)
    mu_CRGD, dt_CRGD, m_CRGD, vc_CRGD = tuning_process(CRGD, 4, n_times, steps, x0, seed=1)

    np.save("mu_CM_ex3c", mu_CM)
    np.save("dt_CM_ex3c", dt_CM)

    np.save("mu_NAG_ex3c", mu_NAG)
    np.save("dt_NAG_ex3c", dt_NAG)

    np.save("mu_RGD_ex3c", mu_RGD)
    np.save("dt_RGD_ex3c", dt_RGD)
    np.save("m_RGD_ex3c", m_RGD)
    np.save("vc_RGD_ex3c", vc_RGD)

    np.save("mu_CRGD_ex3c", mu_CRGD)
    np.save("dt_CRGD_ex3c", dt_CRGD)
    np.save("m_CRGD_ex3c", m_CRGD)
    np.save("vc_CRGD_ex3c", vc_CRGD)


def find_params(n_times, num_trials, steps_trial, method, flag):
    '''
    In this function, we tuned the parameters for Figure 4d. We run the function
    tuning_process() for each initial condition x0.
    '''
    Mu = []
    Dt = []
    M = []
    Vc = []
    for i in range(num_trials):
        np.random.seed(i)
        x0 = np.random.uniform(-5, 5, 2)
        mu, dt, m, vc = tuning_process(method, flag, n_times, steps_trial, x0, seed=i)
        Mu.append(mu)
        Dt.append(dt)
        M.append(m)
        Vc.append(vc)
    return Mu, Dt, M, Vc


def params_Fig4d(n_times, num_trials, steps_trial):
    '''
    In this function, we called the tuned parameters for Figure 4d.
    '''
    mu_CM, dt_CM, _, _ = find_params(n_times, num_trials, steps_trial, CM, 1)
    mu_NAG, dt_NAG, _, _ = find_params(n_times, num_trials, steps_trial, NAG, 2)
    mu_RGD, dt_RGD, m_RGD, vc_RGD = find_params(n_times, num_trials, steps_trial, RGD, 3)
    mu_CRGD, dt_CRGD, m_CRGD, vc_CRGD = find_params(n_times, num_trials, steps_trial, CRGD, 4)

    mu_CM_mean, dt_CM_mean = np.mean(mu_CM), np.mean(dt_CM)
    mu_NAG_mean, dt_NAG_mean = np.mean(mu_NAG), np.mean(dt_NAG)
    mu_RGD_mean, dt_RGD_mean, m_RGD_mean, vc_RGD_mean = np.mean(mu_RGD), np.mean(dt_RGD), np.mean(m_RGD), np.mean(vc_RGD)
    mu_CRGD_mean, dt_CRGD_mean, m_CRGD_mean, vc_CRGD_mean = np.mean(mu_CRGD), np.mean(dt_CRGD), np.mean(m_CRGD), np.mean(vc_CRGD)

    np.save("mu_CM_ex3d", mu_CM_mean)
    np.save("dt_CM_ex3d", dt_CM_mean)

    np.save("mu_NAG_ex3d", mu_NAG_mean)
    np.save("dt_NAG_ex3d", dt_NAG_mean)

    np.save("mu_RGD_ex3d", mu_RGD_mean)
    np.save("dt_RGD_ex3d", dt_RGD_mean)
    np.save("m_RGD_ex3d", m_RGD_mean)
    np.save("vc_RGD_ex3d", vc_RGD_mean)

    np.save("mu_CRGD_ex3d", mu_CRGD_mean)
    np.save("dt_CRGD_ex3d", dt_CRGD_mean)
    np.save("m_CRGD_ex3d", m_CRGD_mean)
    np.save("vc_CRGD_ex3d", vc_CRGD_mean)


# -------- Tuning process for example 3 ---------
# To simulate  parameters for Example 3, uncommented the following lines

# params_Fig4b(steps=300, n_times=1500)
# params_Fig4c(steps=300, n_times=1500)
# params_Fig4d(n_times = 100, num_trials = 100, steps_trial = 100)

