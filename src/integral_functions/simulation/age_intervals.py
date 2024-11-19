import numpy as np
from numpy.typing import NDArray

from integral_functions.typing import Numeric


def tight_intervals_generator(
    age_start: Numeric,
    age_end: Numeric,
    num_groups: Numeric,
    age_interval_ub: Numeric = 5,
) -> NDArray:
    age_starts = np.random.uniform(
        low=age_start,
        high=age_end - age_interval_ub,
        size=num_groups,
    )
    age_intervals = np.random.uniform(
        low=0.0, high=age_interval_ub, size=num_groups
    )
    age_ends = age_starts + age_intervals
    age_ranges = np.vstack([age_starts, age_ends])

    return age_ranges


def intervals_generator(
    age_start: Numeric, age_end: Numeric, num_groups: Numeric
) -> NDArray:
    age_ranges = np.random.uniform(
        low=age_start, high=age_end, size=2 * num_groups
    ).reshape((2, num_groups))
    age_ranges.sort(axis=0)
    return age_ranges


def bounded_intervals_generator(
    age_start: Numeric,
    age_end: Numeric,
    num_groups: Numeric,
    age_interval_lb: Numeric = 0,
    age_interval_ub: Numeric = 5,
) -> NDArray:
    age_starts = np.random.uniform(
        low=age_start,
        high=age_end - age_interval_ub,
        size=num_groups,
    )
    age_intervals = np.random.uniform(
        low=age_interval_lb, high=age_interval_ub, size=num_groups
    )
    age_ends = age_starts + age_intervals
    age_ranges = np.vstack([age_starts, age_ends])

    return age_ranges


def int_bounded_intervals_generator(
    age_start: Numeric,
    age_end: Numeric,
    num_groups: Numeric,
    age_interval_lb: Numeric = 0,
    age_interval_ub: Numeric = 5,
) -> NDArray:
    age_starts = np.random.randint(
        low=age_start,
        high=age_end - age_interval_ub,
        size=num_groups,
    )
    age_intervals = np.random.randint(
        low=age_interval_lb, high=age_interval_ub, size=num_groups
    )
    age_ends = age_starts + age_intervals
    age_ranges = np.vstack([age_starts, age_ends])

    return age_ranges


def pop_density_generator(
    lb: Numeric,
    ub: Numeric,
    population_densities: NDArray,
    grid_points: NDArray,
) -> NDArray:
    pop_densities_lb = population_densities[grid_points >= lb]
    pop_densities_ub = population_densities[grid_points <= ub]
    intersection = np.intersect1d(pop_densities_lb, pop_densities_ub)
    intersection_correct_order = pop_densities_lb[
        np.isin(pop_densities_lb, intersection)
    ]
    return intersection_correct_order
