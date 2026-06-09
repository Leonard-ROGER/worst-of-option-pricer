import numpy as np
import matplotlib.pyplot as plt
from .pricer import pricer_worst_of_antithetic

def plot_convergence(S0, r, sigma, T, K, N, correlation_matrix, option_type, n_point, starting_point, ending_point):
    """
    Plot the convergence of the Monte Carlo price estimate as the number of simulations evolve

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
    option_type : str
        Type of option, either "call" or "put"
    n_point : int
        Number of simulation counts to evaluate
    starting_point : float
        Lower bound of the simulation range
    ending_point : float
        Upper bound of the simulation range
    """

    n_simulations_array = np.logspace(starting_point, ending_point, n_point)

    price_array = np.zeros(len(n_simulations_array))
    confidence_interval_up_array = np.zeros(len(n_simulations_array))
    confidence_interval_down_array = np.zeros(len(n_simulations_array))

    for i in range(len(n_simulations_array)):
        price_array[i], _, (confidence_interval_down_array[i], confidence_interval_up_array[i]) = pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_matrix, int(n_simulations_array[i]), option_type)

    plt.figure()
    plt.plot(n_simulations_array, price_array, label="Price")
    plt.fill_between(n_simulations_array, confidence_interval_down_array, confidence_interval_up_array, label="Confidence Intervals", alpha=0.5)
    plt.xscale('log')
    plt.xlabel("Number of simulations")
    plt.ylabel("Worst-of " + option_type + " price")
    plt.title("Monte Carlo Convergence Worst-of " + option_type + " pricing")
    plt.legend()
    plt.show()

def plot_correlation(S0, r, sigma, T, K, N, n_simulations, n_point):
    """
    Plot the Worst-of Call and Put prices as a function of the correlation.

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
    n_simulations : int
        Number of Monte Carlo simulations per pricing call
    n_point : int
        Number of correlation values to evaluate between -0.99 and 0.99
    """

    correlation_array = np.linspace(-0.99, 0.99, n_point)
    correlation_matrix = np.zeros((len(S0),len(S0)))

    price_put_array = np.zeros(n_point)
    price_call_array = np.zeros(n_point)

    for x in range(n_point):

        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix[i])):
                if i != j:
                    correlation_matrix[i][j] = correlation_array[x]

                else : correlation_matrix[i][j] = 1

        price_put_array[x] = pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_matrix,n_simulations, "put") [0]
        price_call_array[x] = pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_matrix, n_simulations, "call")[0]

    plt.figure()

    plt.plot(correlation_array, price_put_array, label="Worst-of Put Price", color="red")
    plt.plot(correlation_array, price_call_array, label="Worst-of Call Price", color="blue")

    plt.legend()
    plt.xlabel("Mean Correlation between underlying assets")
    plt.ylabel("Price of Worst-of Options")
    plt.title("Monte Carlo Worst-of Options Price as a Function of Correlation")
    plt.show()