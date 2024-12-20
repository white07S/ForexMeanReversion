import numpy as np
from scipy.optimize import minimize
from .ou_params import discrete_ou_moments

def ou_log_likelihood(params, x, dt):
    # params = [theta, mu, sigma]
    theta, mu, sigma = params
    if sigma <= 0 or theta <= 0:
        return np.inf
    # Compute log-likelihood of OU data
    # x_t ~ Normal(mean, var) from OU discretization
    n = len(x)
    ll = 0
    for i in range(1, n):
        mean, var = discrete_ou_moments(x[i-1], theta, mu, sigma, dt)
        ll += -0.5*(np.log(2*np.pi*var) + (x[i]-mean)**2/var)
    return -ll # negative log-likelihood for minimization

def estimate_ou_params_mle(x, dt):
    # Initial guesses
    x_mean = np.mean(x)
    x_var = np.var(x)
    # heuristic init
    theta_init = 0.01
    mu_init = x_mean
    sigma_init = np.sqrt(2*theta_init*x_var)
    init_params = [theta_init, mu_init, sigma_init]

    bounds = [(1e-6, None), (None, None), (1e-6, None)]
    res = minimize(ou_log_likelihood, init_params, args=(x, dt), bounds=bounds, method='L-BFGS-B')
    if res.success:
        return res.x
    else:
        raise ValueError("OU parameter estimation did not converge.")

def rolling_ou_estimation(x, dt, window):
    # For each endpoint after 'window', estimate OU parameters
    params_list = []
    times = []
    for i in range(window, len(x)):
        segment = x[i-window:i]
        theta, mu, sigma = estimate_ou_params_mle(segment, dt)
        params_list.append((theta, mu, sigma))
        times.append(i)
    return np.array(times), np.array(params_list)