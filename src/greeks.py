import numpy as np
from .pricer import pricer_worst_of_antithetic

#Explanation about greeks computation with Bump and Revalue method and how to choose h for each greek : https://equicurious.com/learn/derivatives/derivative-pricing-and-models/estimating-greeks-numerically

def greeks_computation(S0, r, sigma, T, K, N, correlation_matrix, n_simulations, option_type):
    """
    Compute the first-order Greeks and Gamma of a Worst-of Option via bump-and-revalue

    Re-prices the option with each parameter bumped up and down by h, using fixed seed for option pricing
    to delete Monte Carlo noise.

    Parameters
    ----------
    S0 : np.ndarray of shape (n_underlying,)
        Initial spot prices for each underlying asset at t=0
    r : float
        Risk-free interest rate
    sigma : np.ndarray of shape (n_underlying,)
        Implied volatility for each underlying asset
    T : float
        Time to maturity in years
    K : float
        Strike as a performance ratio (1.0 = ATM)
    N : float
        Fixed notional amount
    correlation_matrix : np.ndarray of shape (n_underlying, n_underlying)
        Correlation matrix between the n underlying assets
    n_simulations : int
        Number of Monte Carlo simulations
    option_type : str
        Type of option, either "call" or "put"

    Returns
    -------
        - "delta" : np.ndarray of shape (n_underlying,)
        - "gamma" : np.ndarray of shape (n_underlying,)
        - "vega"  : np.ndarray of shape (n_underlying,)
        - "theta" : float
        - "rho_rate" : float
        - "rho_correlation" : float
    """

    h_vega = 0.01 # h is equal to 1 vol point
    h_theta = 1/365 # h is equal to 1 calendar day
    h_rho_rate = 0.0001 #h is equal to 1 bp
    h_rho_correlation = 0.01 # WARNING: correlation coefficients need to stay between -0.99 and 0.99

    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix[i])):
            if i != j:
                if correlation_matrix[i][j]+ h_rho_correlation > 1 or correlation_matrix[i][j]- h_rho_correlation < -1:
                    raise ValueError("Correlation coefficients need to stay between -0.99 and 0.99")


    seed = np.random.randint(1000000000)

    #Delta computation

    delta = np.zeros(len(S0))

    for i in range(len(S0)):

        h_delta = S0[i] * 0.01

        S0_up = np.ndarray.copy(S0)
        S0_up[i] += h_delta

        S0_down = np.ndarray.copy(S0)
        S0_down[i] -= h_delta

        np.random.seed(seed)
        price_up = pricer_worst_of_antithetic(S0_up, r, sigma, T, K, N, correlation_matrix, n_simulations, option_type)[0]

        np.random.seed(seed)
        price_down = pricer_worst_of_antithetic(S0_down, r, sigma, T, K, N, correlation_matrix, n_simulations, option_type)[0]

        delta[i] = (price_up-price_down)/ (2 * h_delta)


    #Gamma computation

    gamma = np.zeros(len(S0))

    for i in range(len(S0)):
        h_gamma = S0[i] * 0.01

        S0_up = np.ndarray.copy(S0)
        S0_up[i] += h_gamma

        S0_down = np.ndarray.copy(S0)
        S0_down[i] -= h_gamma

        np.random.seed(seed)
        price = pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_matrix, n_simulations, option_type)[0]

        np.random.seed(seed)
        price_up = pricer_worst_of_antithetic(S0_up, r, sigma, T, K, N, correlation_matrix, n_simulations, option_type)[0]

        np.random.seed(seed)
        price_down = pricer_worst_of_antithetic(S0_down, r, sigma, T, K, N, correlation_matrix, n_simulations, option_type)[0]

        gamma[i] = (price_up - (2 * price) + price_down) / (h_gamma ** 2)


    #Vega computation

    vega = np.zeros(len(S0))

    for i in range(len(S0)):

        sigma_up = np.ndarray.copy(sigma)
        sigma_up[i] += h_vega

        sigma_down = np.ndarray.copy(sigma)
        sigma_down[i] -= h_vega

        np.random.seed(seed)
        price_up = pricer_worst_of_antithetic(S0, r, sigma_up, T, K, N, correlation_matrix, n_simulations, option_type)[0]

        np.random.seed(seed)
        price_down = pricer_worst_of_antithetic(S0, r, sigma_down, T, K, N, correlation_matrix, n_simulations, option_type)[0]

        vega[i] = (price_up-price_down)/ (2 * h_vega)


    #Theta computation

    np.random.seed(seed)
    price_down = pricer_worst_of_antithetic(S0, r, sigma, T - h_theta, K, N, correlation_matrix, n_simulations, option_type)[0]

    np.random.seed(seed)
    price = pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_matrix, n_simulations, option_type)[0]

    theta = (price_down - price) / h_theta #There is not a price_up in formula because we can't go back in time


    #Rho_rate computation

    np.random.seed(seed)
    price_up = pricer_worst_of_antithetic(S0, r + h_rho_rate, sigma, T, K, N, correlation_matrix, n_simulations, option_type)[0]

    np.random.seed(seed)
    price_down = pricer_worst_of_antithetic(S0, r - h_rho_rate, sigma, T, K, N, correlation_matrix, n_simulations, option_type)[0]

    rho_rate = (price_up - price_down) / (2 * h_rho_rate)


    #Rho_correlation computation

    correlation_matrix_up = np.ndarray.copy(correlation_matrix)
    correlation_matrix_down = np.ndarray.copy(correlation_matrix)

    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix[i])):
            if i != j:
                correlation_matrix_up[i][j] += h_rho_correlation
                correlation_matrix_down[i][j] -= h_rho_correlation

    np.random.seed(seed)
    price_up = pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_matrix_up, n_simulations, option_type)[0]

    np.random.seed(seed)
    price_down = pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_matrix_down, n_simulations, option_type)[0]

    rho_correlation = (price_up - price_down) / (2 * h_rho_correlation)


    np.random.seed(None)
    return delta, gamma ,vega ,theta, rho_rate, rho_correlation

# Note: Delta and Gamma are 0 as ST/S0 is independent of S0
# If the product had a strike in euros, Delta & Gamma would be different from zero.
# In this program, the strike is a performance ratio.