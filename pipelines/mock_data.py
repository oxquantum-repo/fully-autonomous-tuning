import numpy as np
from scipy.optimize import curve_fit


def noisy_lorentzian(x, x0, gamma, A, noise_level):
    """Return a noisy lorentzian.

    Args:
        x (np.array): x values
        x0 (float): center of the lorentzian
        gamma (float): width of the lorentzian
        A (float): amplitude of the lorentzian
        noise_level (float): standard deviation of the noise
    """
    return lorentzian(x, x0, gamma, A) + noise_level * np.random.randn(len(x))


def noisy_rabi_oscillations(t, A, omega, tau, phase, noise_level):
    """Return noisy rabi oscillations.

    Args:
        t (np.array): time values
        A (float): amplitude of the oscillations
        omega (float): frequency of the oscillations
        tau (float): decay time of the oscillations
        phase (float): phase of the oscillations
    """
    return rabi_oscillations(t, A, omega, tau, phase) + noise_level * np.random.randn(len(t))


def lorentzian(x, x0, gamma, A):
    """Return a lorentzian."""
    return A * gamma / (np.pi) / ((x - x0) ** 2 + (gamma) ** 2)


def rabi_oscillations(t, A, omega, tau, phase):
    """ Function for Rabi oscillations. """
    return A * np.exp(-t / tau) * np.cos(omega * t + phase)


def fit_rabi_oscillations(t, signal, guess):
    """ Fit Rabi oscillations. """
    # Curve fitting
    popt, pcov = curve_fit(rabi_oscillations, t, signal, p0=guess, maxfev=5000)

    return popt, pcov