import numpy as np

def payoff_worst_of_call(ST,S0,K,N):

    """
        Compute the payoff for a Worst-of-Call Option across all simulations

        The payoff is based on the performance of the worst performing underlying asset : max(min_i(Si(T)/Si(0)) - K, 0) * N

        Parameters
        ----------
        ST : np.ndarray of shape (n_simulations, n_underlying)
            Simulated spot prices for each underlying asset at maturity
        S0 : np.ndarray of shape (n_underlying,)
            Initial spot prices for each underlying asset at t=0
        K : float
            Strike expressed as a ratio (1.0 = ATM)
        N : float
            Fixed notional amount

        Returns
        -------
        np.ndarray of shape (n_simulations,)
            Call payoff for each simulation
        """

    return np.maximum(np.min((ST / S0), axis = 1) - K,0) * N

def payoff_worst_of_put(ST,S0,K,N):

    """
        Compute the payoff for a Worst-of-Put Option across all simulations

        The payoff is based on the performance of the worst performing underlying asset : max(K - min_i(Si(T)/Si(0)), 0) * N

        Parameters
        ----------
        ST : np.ndarray of shape (n_simulations, n_underlying)
            Simulated spot prices for each underlying asset at maturity
        S0 : np.ndarray of shape (n_underlying,)
            Initial spot prices for each underlying asset at t=0
        K : float
            Strike expressed as a ratio (1.0 = ATM)
        N : float
            Fixed notional amount

        Returns
        -------
        np.ndarray of shape (n_simulations,)
            Put payoff for each simulation
        """
    return np.maximum(K - np.min((ST / S0), axis = 1), 0) * N