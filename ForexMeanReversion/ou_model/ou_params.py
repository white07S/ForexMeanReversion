import numpy as np

def ou_mean_revert_level(theta, mu, sigma):
    # Stationary mean of OU is mu
    return mu

def ou_stationary_var(theta, sigma):
    # Stationary variance of OU is sigma^2/(2*theta)
    return (sigma**2)/(2*theta)

def discrete_ou_moments(x_prev, theta, mu, sigma, dt):
    # Mean of X_t given X_(t-1)
    mean = x_prev * np.exp(-theta*dt) + mu*(1 - np.exp(-theta*dt))
    var = (sigma**2/(2*theta))*(1 - np.exp(-2*theta*dt))
    return mean, var