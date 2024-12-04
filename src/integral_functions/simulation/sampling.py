from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from integral_functions.simulation.integrate_functions import (
    integrate_cov,
    integrate_denom,
    integrate_density,
    integrate_expecatation,
)
from integral_functions.typing import Numeric


def _cdf_gen(
    age_start: Numeric,
    age_end: Numeric,
    density: Callable,
    cdf_gridsize: int = 10000,
    age_mid: int = 35,
) -> NDArray:
    age_range = np.linspace(age_start, age_end, cdf_gridsize)
    distribution = np.array([density(age, age_mid) for age in age_range])

    distribution /= np.sum(distribution)
    cdf = np.cumsum(distribution)
    cdf /= cdf[-1]

    return np.vstack([cdf, age_range])


def sample_probability_of_death(
    age_start: Numeric,
    age_end: Numeric,
    sample_size: int,
    density: Callable,
    true_prob: Callable,
    cdf_gridsize: int = 10000,
    prob_args: dict | None = None,
) -> float:
    """Samples sample_size many draws from density and calculates the probability of death by true_prob given the age. Then, according to this probability the samples are chosen to be
    dead or alive according to a Bernoulli draw with parameter p = true_prob. It then finds the average number of dead observations.

    Args:
        age_start (float): Lower bound on the age interval of interest.
        age_end (float): Upper bound on the age interval of interest.
        sample_size (int): Number of age samples in the age interval of interest.
        density (Callable): Probability density function of the distribution of ages for the population.
        true_prob (Callable): Death rate function that takes as input an age and calculates the probability of death given the age.
        cdf_gridsize (int, optional): Number of grid points given to construct the density and CDF. Defaults to 10000.

    Returns:
        float: The average number of samples who are chosen binomially to have died.
    """
    sample_size = int(sample_size)
    age_cdf = _cdf_gen(age_start, age_end, density, cdf_gridsize)
    cdf = age_cdf[0, :]
    age_range = age_cdf[1, :]
    samples = np.random.rand(sample_size)
    age_samples = np.interp(samples, cdf, age_range)
    prob_args = prob_args or {}
    prob_death = true_prob(age_samples, **prob_args)
    dead_or_alive = np.random.binomial(n=1, p=prob_death)
    sum_dead_or_alive = np.sum(dead_or_alive)
    avg_dead_or_alive = sum_dead_or_alive / sample_size

    return avg_dead_or_alive


def probability_of_death_no_error(
    age_start: Numeric,
    age_end: Numeric,
    density: Callable,
    true_prob: Callable,
    age_mid: Numeric | None,
    link_function: Callable | None = None,
    # prob_args: dict | None = None,
) -> float:
    """Calculates the average number of dead observations in the age interval of interest (from age_start to age_end).
    No sampling error is incurred since integration is done directly on the relevant functions.

    Parameters
    ----------
    age_start
        Lower bound on the age interval of interest.
    age_end
        Upper bound on the age interval of interest.
    density
        Probability density function of the distribution of ages for the population.
    true_prob
        Death rate function that takes as input an age and calculates the probability of death given the age.
    age_mid
        Knot where the age distribution changes from one function to the next.
    link_function
        The link function used for the death rate function. Assumed to be expit.

    Returns
    -------
    DataFrame
        The dataframe of required data.

    """
    if isinstance(link_function, Callable):
        true_prob = link_function(true_prob)
    if age_mid is None:
        int_exp = integrate_expecatation
        int_dens = integrate_density
        num = int_exp(
            func=true_prob,
            density=density,
            age_start=age_start,
            age_end=age_end,
        )
        denom = int_dens(density=density, age_start=age_start, age_end=age_end)
    else:
        int_exp = integrate_cov
        int_dens = integrate_denom
        num = int_exp(
            func=true_prob,
            density=density,
            age_start=age_start,
            age_end=age_end,
            age_mid=age_mid,
        )
        denom = int_dens(
            density=density,
            age_start=age_start,
            age_end=age_end,
            age_mid=age_mid,
        )

    # prob_args = prob_args or {}

    return num / denom
