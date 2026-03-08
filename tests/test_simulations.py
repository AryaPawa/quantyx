from quantyx.simulations import MonteCarloCall


def test_monte_carlo_call():

    strike = 100
    n = 1000
    r = 0.01
    S = 100
    mu = 0.05
    sigma = 0.2
    dt = 0.01
    T = 1

    option = MonteCarloCall(strike, n, r, S, mu, sigma, dt, T)

    assert option.price > 0