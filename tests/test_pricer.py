import numpy as np
from src.pricer import pricer_worst_of_antithetic

S0 = np.array([100.0, 100.0])
r = 0.05
sigma = np.array([0.2, 0.2])
T = 1.0
K = 1.0
N = 1000
correlation_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
n_simulations = 1000000

S0_single = np.array([100.0])
sigma_single = np.array([0.2])
correlation_single = np.array([[1.0]])

correlation_low  = np.array([[1.0, 0.2], [0.2, 1.0]])
correlation_high = np.array([[1.0, 0.8], [0.8, 1.0]])

def test_call_price_is_positive():
    # A call option price must always be non-negative

    price, _, _ = pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_matrix, n_simulations, "call")
    assert price > 0

def test_worst_of_call_inferior_vanilla_call():
    # A Worst-off call option price must be inferior or equal to a vanilla call with the same variables

    seed = np.random.randint(0,1000000000)

    np.random.seed(seed)
    price_vanilla_call, _, _ = pricer_worst_of_antithetic(S0_single, r, sigma_single, T, K, N,
                                                          correlation_single, n_simulations, "call")

    np.random.seed(seed)
    price_woc, _, _ = pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_matrix, n_simulations, "call")

    np.random.seed(None)
    assert price_vanilla_call >= price_woc

def test_price_increase_with_correlation():
    # A Worst-off call option price must increase when correlation increase too

    seed = np.random.randint(0, 1000000000)

    np.random.seed(seed)
    price_low_correlation, _, _ = pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_low, n_simulations, "call")

    np.random.seed(seed)
    price_high_correlation, _, _ = pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_high, n_simulations,"call")

    np.random.seed(None)
    assert price_low_correlation <= price_high_correlation