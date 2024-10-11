from typing import Callable

import numpy as np
from numpy.typing import NDArray


def midpoint_to_disc(
    discpoint: int | float, gridpoint: int | float
) -> int | float:
    r"""Given a discretization point :math:`x_i` and a grid point math:`g_i`,
    this function calculates the following point via the formula.
    .. math::

        x_{i+1} = 2g_i - x_i

    The use case of this formulation applies to integration via midpoint
    quadrature rules. In this way, the grid point is considered to be the
    midpoint as it falls exactly in the middle of the discretization points
    :math:`x_i` and :math:`x_{i+1}`.

    Parameters
    ----------
    discpoint
        The preceding discretization point used to calculate the following
        point.
    gridpoint
        The midpoint used for calculating the next discretization point.

    Returns
    -------
    int | float
        The next discretization point in the sequence.

    """
    # assert gridpoint > discpoint
    return 2 * gridpoint - discpoint


def get_discretizations(
    L: int | float,
    U: int | float,
    grid_points: NDArray,
    grid_to_disc: Callable = midpoint_to_disc,
) -> NDArray:
    r"""Creates appropriate discretizations from grid points. Effectively, this
    function will generate points such that every pair of adjacent elements in
    the returned vector will have a grid point (from grid_points) between it.
    The method of point generation depends on the callable grid_to_disc.

    Parameters
    ----------
    L
        Lower bound on the age range of interest.
    U
        Upper bound on the age range of interest.
    grid_points
        The n x 1 vector of points that serve as the evaluation points of the
        function we seek to model. The evaluation is done at inference time in
        solving for :math:`\theta`. We use the grid points to build age bins.
    grid_to_disc
        Function to map a grid point and the immediately prior discretization
        point into a discretization point.

    Returns
    -------
    NDArray
        The (n+1) x 1 vector of discretization points along which we create our
        age bins.

    """
    grid_points = np.asarray(grid_points)
    grid_len = grid_points.shape[0]

    disc_list = [grid_to_disc(L, grid_points[0])]
    for i in range(1, grid_len):
        discpoint = disc_list[i - 1]
        gridpoint = grid_points[i]
        disc_list.append(grid_to_disc(discpoint, gridpoint))
    disc_list.insert(0, L)
    assert disc_list[-1] == U

    return np.array(disc_list)
