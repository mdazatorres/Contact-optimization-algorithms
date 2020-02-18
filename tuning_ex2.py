
import numpy as np
import examples as exa
from integrators import CM, RGD, NAG, CRGD


def tuning_process(method, seed, flag, n_times, steps_trial):
    '''
    In this function, we do the random parameter search for each method (CM, NAG, RGD, CRGD).
    '''
    min_f = 1e+16
    mu_new = 0
    dt_new = 0
    m_new = 0
    vc_new = 0
    ex = exa.Example2(seed)

    for i in range(n_times):
        # -------------    CM    -------------
        if flag == 1:
            dt = np.random.uniform(1e-4, 1e-2)
            mu = np.random.uniform(0.6, 0.95)
            ite_x = method(ex, mu, dt, steps_trial)
            m = 0
            vc = 0
        # -------------    NAG    -------------
        elif flag == 2:
            dt = np.random.uniform(1e-4, 1e-2)
            mu = np.random.uniform(0.6, 0.95)
            ite_x = method(ex, mu, dt, steps_trial)
            m = 0
            vc = 0
        # -------------    RGD    -------------
        elif flag == 3:
            dt = np.random.uniform(1e-4, 8*1e-3)
            mu = np.random.uniform(0.6, 0.8)
            vc = np.random.uniform(1e+3, 1e+5)
            m = np.random.uniform(1e-6, 1e-4)
            ite_x = method(ex, mu, dt, vc, m, steps_trial)
        # -------------    CRGD    -------------
        else:
            dt = np.random.uniform(1e-4, 8 * 1e-3)
            mu = np.random.uniform(0.65, 0.95)
            vc = np.random.uniform(1e+3, 1e+5)
            m = np.random.uniform(1e-4, 1e-2)
            ite_x = method(ex, [vc, m, mu], dt, steps_trial)

        f_sim = np.apply_along_axis(ex.f, 1, ite_x)
        min_fnew = min(f_sim)
        print('minfnew', min_fnew, 'mu', mu, 'dt', dt, 'm', m, 'vc', vc, 'flag', flag)

        if min_fnew < min_f:
            min_f = min_fnew
            mu_new = mu
            dt_new = dt
            m_new = m
            vc_new = vc

    return mu_new, dt_new, m_new, vc_new


def find_params(n_times, num_trials, steps_trial, method, flag):
    '''
    In this function, we tuned the parameters for Figure 3. We run the function
    tuning_process() for each initial condition x0.
    '''
    Mu = []
    Dt = []
    M = []
    Vc = []
    for i in range(num_trials):
        mu, dt, m, vc = tuning_process(method, i, flag, n_times, steps_trial)
        Mu.append(mu)
        Dt.append(dt)
        M.append(m)
        Vc.append(vc)

    return Mu, Dt, M, Vc


def params_Fig3(n_times, num_trials, steps_trial):
    '''
    In this function, we called the tuned parameters for Figure 3.
    '''
    mu_CM, dt_CM, _, _ = find_params(n_times, num_trials, steps_trial, CM, 1)
    mu_NAG, dt_NAG, _, _ = find_params(n_times, num_trials, steps_trial, NAG, 2)
    mu_RGD, dt_RGD, m, vc = find_params(n_times, num_trials, steps_trial, RGD, 3)
    mu_CRGD, dt_CRGD, m_CRGD, vc_CRGD = find_params(n_times, num_trials, steps_trial, CRGD, 4)

    mu_CM_mean, dt_CM_mean = np.mean(mu_CM), np.mean(dt_CM)
    mu_NAG_mean, dt_NAG_mean = np.mean(mu_NAG), np.mean(dt_NAG)
    mu_RGD_mean, dt_RGD_mean, m_mean, vc_mean = np.mean(mu_RGD), np.mean(dt_RGD), np.mean(m), np.mean(vc)
    mu_CRGD_mean, dt_CRGD_mean, m_CRGD_mean, vc_CRGD_mean = np.mean(mu_CRGD), np.mean(dt_CRGD), np.mean(m_CRGD), np.mean(vc_CRGD)

    np.save("mu_CM_ex2", mu_CM_mean)
    np.save("dt_CM_ex2", dt_CM_mean)

    np.save("mu_NAG_ex2", mu_NAG_mean)
    np.save("dt_NAG_ex2", dt_NAG_mean)

    np.save("mu_RGD_ex2", mu_RGD_mean)
    np.save("dt_RGD_ex2", dt_RGD_mean)
    np.save("m_RGD_ex2", m_mean)
    np.save("vc_RGD_ex2", vc_mean)

    np.save("mu_CRGD_ex2", mu_CRGD_mean)
    np.save("dt_CRGD_ex2", dt_CRGD_mean)
    np.save("m_CRGD_ex2", m_CRGD_mean)
    np.save("vc_CRGD_ex2", vc_CRGD_mean)

# -------- Tuning process for example 2 ---------
# To simulate  parameters for Example 2, uncommented the following line
# params_Fig3(n_times = 100, num_trials = 50, steps_trial = 300)







