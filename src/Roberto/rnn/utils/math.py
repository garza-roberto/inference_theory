def jacobian(rates, weights, taus = np.ones(4)):
    n_pop = weights.shape[0]
    phi_prime_mat = np.diag(phi_derivative(rates))
    T_inv = np.diag(1/taus)
    jacobian = T_inv@((phi_prime_mat @ weights) - np.eye(n_pop))
    return jacobian