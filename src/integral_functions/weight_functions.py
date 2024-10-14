import numpy as np
from numpy.typing import NDArray

from integral_functions.methods.inducing_points import get_discretizations
from integral_functions.typing import Numeric
from integral_functions.vectorized_funcs import build_indices_midpoint


def build_integration_weights_midpoint(
    lb: NDArray, ub: NDArray, grid_points: NDArray
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
    val[end_points[:-1]] = grid_points[lb_index + 1] - lb
    val[end_points[1:] - 1] = ub - grid_points[ub_index - 1]

    return (val, (row_index, col_index))


def get_weights(
    lb: Numeric,
    ub: Numeric,
    population_density: NDArray,
    grid_points: NDArray,
) -> NDArray:
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
    discretizations = get_discretizations(lb, ub, grid_points)
    age_bin_lengths = np.diff(discretizations)
    weights = population_density * age_bin_lengths

    return weights
