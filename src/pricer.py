import numpy as np
from .simulation import simulate_st_vectorized, simulate_st_antithetic
from .payoff import payoff_worst_of_call, payoff_worst_of_put


def pricer_worst_of(S0, r, sigma, T, K, N, correlation_matrix, n_simulations, option_type):

    """
        Price a Worst-of Option using Monte Carlo simulation

        Simulates correlated asset price via Cholesky, computes payoffs, and returns the Monte Carlo price with its standard error and 95% confidence interval

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
        price : float
            Option price
        standard_error : float
            Standard error
        confidence_interval : tuple of (float, float)
            95% confidence interval (lower bound, upper bound)
        """

    ST = simulate_st_vectorized(S0, r, sigma, T, correlation_matrix, n_simulations)

    if option_type == "call":
        payoff = payoff_worst_of_call(ST,S0,K,N)

    elif option_type == "put":
        payoff = payoff_worst_of_put(ST,S0,K,N)

    else:
        raise ValueError("Invalid option type")

    price = np.exp(-r * T) * (1 / n_simulations) * np.sum(payoff)

    standard_error = np.std(payoff) / np.sqrt(n_simulations) * np.exp(-r * T)

    confidence_interval = (float(price - 1.96 * standard_error), float(price + 1.96 * standard_error))

    return float(price), float(standard_error), confidence_interval

def pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_matrix, n_simulations, option_type):

    """
        Price a Worst-of Option using Monte Carlo simulation with Variance Reduction using Antithetic Values

        Simulates correlated asset price via Cholesky, computes payoffs, and returns the Monte Carlo price with its standard error and 95% confidence interval

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
        price : float
            Option price
        standard_error : float
            Standard error
        confidence_interval : tuple of (float, float)
            95% confidence interval (lower bound, upper bound)
        """

    ST = simulate_st_antithetic(S0, r, sigma, T, correlation_matrix, n_simulations)

    if option_type == "call":
        payoff = payoff_worst_of_call(ST,S0,K,N)

    elif option_type == "put":
        payoff = payoff_worst_of_put(ST,S0,K,N)

    else:
        raise ValueError("Invalid option type")

    price = np.exp(-r * T) * (1 / n_simulations) * np.sum(payoff)

    payoff, payoff_antithetic = np.split(payoff,2)

    mean_payoff = (payoff+payoff_antithetic)/2

    standard_error = np.std(mean_payoff) / np.sqrt(n_simulations//2) * np.exp(-r * T)

    confidence_interval = (float(price - 1.96 * standard_error), float(price + 1.96 * standard_error))

    return float(price), float(standard_error), confidence_interval