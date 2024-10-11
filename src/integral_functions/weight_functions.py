import numpy as np
from numpy.typing import NDArray

from integral_functions.methods.inducing_points import get_discretizations


def get_weights(
    L: int | float,
    U: int | float,
    population_density: NDArray,
    grid_points: NDArray,
) -> NDArray:
    r"""Function that accepts a valid range of ages [L, U], a vector of
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

    where $p_i$ is the $i$th element of the population_density vector, and the
    discretizations vector is obtained from the get_discretizations function,
    which builds age bins around the grid_points argument.

    Parameters
    ----------
    L
        Lower bound on the age range of interest.
    U
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
    discretizations = get_discretizations(L, U, grid_points)
    age_bin_lengths = np.diff(discretizations)
    weights = population_density * age_bin_lengths

    return weights
