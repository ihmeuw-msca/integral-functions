from typing import Callable

import numpy as np
from numpy.typing import NDArray

from integral_functions.config import AGE_MID
from integral_functions.methods.inducing_points import get_discretizations
from integral_functions.simulation.integrate_functions import integrate_denom
from integral_functions.typing import Numeric
from integral_functions.vectorized_funcs import build_indices_midpoint


def build_integration_weights_midpoint(
    lb: Numeric | NDArray, ub: Numeric | NDArray, grid_points: NDArray
) -> tuple[NDArray, tuple[NDArray, NDArray]]:
    """Compute the integration weights on the grid with midpoint rule.

    Assumptions
    -----------
    We do not explicitly check the following assumptions for efficiency. But
    they should be check before calling this function.

    * The values in `grid_points` are sorted in ascending order and are unique.
    * `lb` is strictly less than `ub`.
    * `lb` is greater than the first value in `grid_points`.
    * `ub` is less than the last value in `grid_points`.

    Parameters
    ----------
    lb
        Lower bound of the integration interval.
    ub
        Upper bound of the integration interval.
    grid_points
        The grid points used for the integration.

    Returns
    -------
    NDArray
        The integration weights.

    """
    if isinstance(lb, Numeric):
        lb = np.array([lb])
    if isinstance(ub, Numeric):
        ub = np.array([ub])

    lb_index = np.searchsorted(grid_points, lb, side="right") - 1
    ub_index = np.searchsorted(grid_points, ub, side="left")
    sizes = ub_index - lb_index
    diffs = np.diff(grid_points)
    row_index, col_index = build_indices_midpoint(
        lb_index, ub_index, sizes.sum()
    )

    val = diffs[col_index]
    # rewrite the end intervals sizes
    end_points = np.hstack([0, np.cumsum(sizes)])
    val[end_points[:-1]] = np.minimum(grid_points[lb_index + 1], ub) - lb
    val[end_points[1:] - 1] = ub - np.maximum(lb, grid_points[ub_index - 1])

    return (val, (row_index, col_index))


def get_weights_densities(
    lb: NDArray,
    ub: NDArray,
    age_density: Callable,
    grid_points: NDArray,
    low_age: Numeric = 0,
    high_age: Numeric = 95,
) -> tuple[NDArray, NDArray]:
    r"""Function that accepts a valid range of ages [lb, ub], a vector of
    population densities, and a vector of midpoints and generates the weights
    used in a numerical integration via midpoint quadrature rules. Importantly,
    this function does not return the numerically integrated values, only the
    weights; i.e., standard numerical integrators solve integration by

    .. math::

        \int_a^b f(x)\mathrm{d}x \approx \sum_{i=0}^n w_i f(x_i)

    and this function merely returns the weights :math:`(w_1, \dots, w_n)`.
    In mortality estimation, the weights are comprised as follows:

    .. math::

        w_i = p_i(discretizations_{i+1} - discretizations_{i})

    where :math:`p_i` is the :math:`i`th element of the population_density
    vector, and the discretizations vector is obtained from the
    `get_discretizations` function, which builds age bins around the
    `grid_points` argument.

    Parameters
    ----------
    lb
        Lower bound on the age range of interest.
    ub
        Upper bound on the age range of interest.
    population_density
        The n x 1 vector of population densities per age bin. In mortality
        estimation, this may be viewed as the proportion dead of the total
        population per age bin.
    grid_points
        The n x 1 vector of points that serve as the evaluation points of the
        function we seek to model. In other words, the grid_points are the
        collection of points :math:`(x_1, \dots, x_n)` in the equation above.
        The evaluation is done at inference time in solving for :math:`\theta`.
        We use the grid points to build age bins.

    Returns
    -------
    NDArray
        A vector of the weights :math:`w_i` for :math:`i=1,\dots, n`.

    """
    discretizations = get_discretizations(
        lb=low_age, ub=high_age, grid_points=grid_points
    )
    age_bin_lengths, idxs = build_integration_weights_midpoint(
        lb=lb, ub=ub, grid_points=discretizations
    )
    discretizations_restrict = discretizations[idxs[1]]
    grid_points_restrict = get_points_in_interval(
        lb=lb, ub=ub, points_to_restrict=grid_points, get_grid_points=True
    )
    # This fixes if you start at a low enough lb
    if grid_points_restrict[0] > grid_points_restrict[1]:
        grid_points_restrict = grid_points_restrict[1:]
    # This fixes when a grid point is lower than your lowest age bin group
    if grid_points_restrict[0] < discretizations_restrict[0]:
        grid_points_restrict = grid_points_restrict[1:]
    discretizations_restrict[0] = lb
    discretizations_restrict = np.append(discretizations_restrict, ub)
    population_density = get_interval_population_density(
        age_density=age_density, discretizations=discretizations_restrict
    )
    # This fixes when your highest grid point is too high
    if grid_points_restrict.shape[0] != population_density.shape[0]:
        grid_points_restrict = grid_points_restrict[:-1]

    weights = population_density * age_bin_lengths

    return (grid_points_restrict, weights, population_density)


def get_points_in_interval(
    lb: NDArray,
    ub: NDArray,
    points_to_restrict: NDArray,
    get_grid_points: bool = False,
) -> NDArray:
    """Returns the grid points restriced to the interval between lb and ub.
    We want to return this since we can then evaluate the functions on only these
    grid points and assume the function evaluations on the grid points outside
    the interval go to zero.

    Parameters
    ----------
    lb
        Lower bound of the integration interval.
    ub
        Upper bound of the integration interval.
    grid_points
        The grid points used for the integration.

    Returns
    -------
    NDArray
        The grid points restricted to the interval of interest.

    """
    _, idxs = build_integration_weights_midpoint(
        lb=lb, ub=ub, grid_points=points_to_restrict
    )
    _, col_index = idxs
    if get_grid_points:
        col_index = np.append(arr=col_index, values=(col_index[-1] + 1))
    # col_index += 1
    return points_to_restrict[col_index]


def get_interval_population_density(
    age_density: Callable,
    discretizations: NDArray,
) -> NDArray:
    """Returns the population restriced to the interval between lb and ub.
    We want to return this since we can then evaluate the functions on only these
    grid points and assume the function evaluations on the grid points outside
    the interval go to zero.

    Parameters
    ----------
    lb
        Lower bound of the integration interval.
    ub
        Upper bound of the integration interval.
    grid_points
        The grid points used for the integration.

    Returns
    -------
    NDArray
        The grid points restricted to the interval of interest.

    """
    pop_list = []
    for i in range(1, len(discretizations)):
        integral_over_interval = integrate_denom(
            density=age_density,
            age_start=discretizations[i - 1],
            age_end=discretizations[i],
            age_mid=AGE_MID,
        )
        pop_list.append(integral_over_interval)
    pop_dens = np.array(pop_list)
    return pop_dens
