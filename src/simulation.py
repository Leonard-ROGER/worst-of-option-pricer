import numpy as np

def simulate_st_naive(S0, r, sigma, T, correlation_matrix, n_simulations):

    """
    Simulate spot prices at t=T for n_underlying correlated underlying assets using Black-Scholes Merton dynamics and Cholesky decomposition

    Parameters
    ----------
    S0 : np.ndarray of shape (n_underlying,)
        Initial spot prices for each underlying asset at t=0
    r : float
        Risk-free interest rate.
    sigma : np.ndarray of shape (n_underlying,)
        Implied volatility for each underlying asset
    T : float
        Time to maturity in years
    correlation_matrix : np.ndarray of shape (n_underlying, n_underlying)
        Correlation matrix between the n underlying assets
    n_simulations : int
        Number of Monte Carlo simulations.

    Returns
    -------
    np.ndarray of shape (n_simulations, n_underlying)
        Simulated spot prices at maturity
        Each row is one simulation each column is one underlying asset simulation.
    """

    ST = np.zeros((n_simulations,len(S0)))
    L = np.linalg.cholesky(correlation_matrix)

    for i in range(n_simulations):

        epsilon = np.random.normal(0, 1, len(S0))
        Z = np.dot(L,epsilon)

        for j in range(len(S0)):

            ST[i][j] = S0[j] * np.exp((r - (sigma[j] ** 2 / 2)) * T + sigma[j] * np.sqrt(T) * Z[j])

    return ST

def simulate_st_vectorized(S0, r, sigma, T, correlation_matrix, n_simulations):

    """
    Simulate spot prices at t=T for n_underlying correlated underlying assets using Black-Scholes Merton dynamics and Cholesky decomposition

    In this function the complexity is heavily lowered compared to simulate_st_naive

    Parameters
    ----------
    S0 : np.ndarray of shape (n_underlying,)
        Initial spot prices for each underlying asset at t=0
    r : float
        Risk-free interest rate.
    sigma : np.ndarray of shape (n_underlying,)
        Implied volatility for each underlying asset
    T : float
        Time to maturity in years
    correlation_matrix : np.ndarray of shape (n_underlying, n_underlying)
        Correlation matrix between the n underlying assets
    n_simulations : int
        Number of Monte Carlo simulations.

    Returns
    -------
    np.ndarray of shape (n_simulations, n_underlying)
        Simulated spot prices at maturity
        Each row is one simulation, each column is one underlying asset simulation
    """

    L = np.linalg.cholesky(correlation_matrix)
    epsilon = np.random.normal(0, 1, (n_simulations,len(S0)))
    Z = epsilon @ L.T

    ST = S0*np.exp((r - (sigma ** 2 / 2)) * T + sigma * np.sqrt(T) * Z)

    return ST

def simulate_st_antithetic(S0, r, sigma, T, correlation_matrix, n_simulations):

    """
    Simulate spot prices at t=T for n_underlying correlated underlying assets using Black-Scholes Merton dynamics,
    Cholesky decomposition and Monte Carlo Variance Reduction using Antithetic Values

    Parameters
    ----------
    S0 : np.ndarray of shape (n_underlying,)
        Initial spot prices for each underlying asset at t=0
    r : float
        Risk-free interest rate.
    sigma : np.ndarray of shape (n_underlying,)
        Implied volatility for each underlying asset
    T : float
        Time to maturity in years
    correlation_matrix : np.ndarray of shape (n_underlying, n_underlying)
        Correlation matrix between the n underlying assets
    n_simulations : int
        Number of Monte Carlo simulations.

    Returns
    -------
    np.ndarray of shape (n_simulations, n_underlying)
        Simulated spot prices at maturity
        Each row is one simulation, each column is one underlying asset simulation
    """

    L = np.linalg.cholesky(correlation_matrix)

    epsilon = np.random.normal(0, 1, (n_simulations//2,len(S0)))

    epsilon = np.concatenate((epsilon, -epsilon), axis=0)

    Z = epsilon @ L.T

    ST = S0*np.exp((r - (sigma ** 2 / 2)) * T + sigma * np.sqrt(T) * Z)

    return ST