from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import expit

from integral_functions.config import c1, c2, c3
from integral_functions.simulation.age_distribution import age_distribution
from integral_functions.simulation.age_intervals import (
    int_bounded_intervals_generator,
)
from integral_functions.simulation.death_rate import (
    expit_death_rate_function,
    func1,
    func2,
    func3,
)
from integral_functions.simulation.sampling import (
    probability_of_death_no_error,
    sample_probability_of_death,
)
from integral_functions.typing import Numeric
from integral_functions.weight_functions import get_weights_densities


def simulation_data_generator(
    low_age: Numeric,
    high_age: Numeric,
    mid_age: Numeric,
    num_groups: Numeric,
    age_interval_lb: Numeric,
    age_interval_ub: Numeric,
    sample_size_low: Numeric,
    sample_size_high: Numeric,
    sample_error: bool,
    prob_args: dict | None = None,
) -> pd.DataFrame:
    """Creates the first three columns of the desired data matrix used for simulation.
    The first column will be a randomly selected (integer) lower bound for an interval.
    The second column will be a randomly selected (integer) upper bound for the interval
    whose lower bound is the corresponding row in the first column.
    The third column is the actual observation.

    Parameters
    ----------
    low_age
        Lowest accepted age.
    high_age
        Highest accepted age.
    num_groups
        Number of rows in the data matrix. Number of intervals we observe.
    age_interval_lb
        Lower bound on the difference between the lower and upper bounds of the integration interval.
    age_interval_ub
        Upper bound on the difference between the lower and upper bounds of the integration interval.
    sample_size_low
        Lower bound for sample size of individuals per study (per row).
    sample_size_high
        Upper bound for sample size of individuals per study (per row).
    prob_args
        Gives the arguments for expit_death_rate_function

    Returns
    -------
    DataFrame
        The dataframe of required data.

    """
    age_ranges = int_bounded_intervals_generator(
        low_age, high_age, num_groups, age_interval_lb, age_interval_ub
    )
    df = pd.DataFrame(
        dict(
            age_start=age_ranges[0],
            age_end=age_ranges[1],
        )
    )

    # sample size for each sample group
    df["sample_size"] = np.random.randint(
        low=sample_size_low, high=sample_size_high, size=num_groups
    )

    if sample_error:
        df["obs"] = df.apply(
            lambda row: sample_probability_of_death(
                row.iloc[0],
                row.iloc[1],
                row.iloc[2],
                age_distribution,
                expit_death_rate_function,
                prob_args=prob_args,
            ),
            axis=1,
        )
    else:
        df["obs"] = df.apply(
            lambda row: probability_of_death_no_error(
                age_start=row.iloc[0],
                age_end=row.iloc[1],
                density=age_distribution,
                true_prob=expit_death_rate_function,
                age_mid=mid_age,
            ),
            axis=1,
        )

    return df


def estimation_dataframe(
    df: pd.DataFrame,
    grid_points: NDArray,
    return_dense_df: bool = False,
    link_function: Callable | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Performs a rudimentary estimation for estimates on the "obs" column in the dataframe generated above.
    This is done through a numerical integration scheme which we simply assume to be midpoint integration.
    Depends strongly on the function gets_weights_densities to construct the necessary weights for integrations
    by the densities.

    Parameters
    ----------
    df
        Dataframe resulting from simulation_data_generator (or that follows this format).
    grid_points
        1D numpy array of midpoints along which we integrate numerically.
    return_dense_df
        Returns dataframe with intermediaray columns used for estimation, along with more simple matrix with just the predicted observations.
    link_function
        The link function used for the death rate function. Assumed to be expit.

    Returns
    -------
    DataFrame | tuple[DataFrame, DataFrame]
        Either a dataframe with just the relevant observation columns, or this dataframe with intermediary columns used for estimation.
    """
    if link_function is None:
        link_function = expit
    df["grids_weights_densities"] = df.apply(
        lambda row: get_weights_densities(
            lb=row.iloc[0],
            ub=row.iloc[1],
            age_density=age_distribution,
            grid_points=grid_points,
        ),
        axis=1,
    )
    df["grid_points"] = df["grids_weights_densities"].apply(lambda x: x[0])
    df["weights"] = df["grids_weights_densities"].apply(lambda x: x[1])
    df["densities"] = df["grids_weights_densities"].apply(lambda x: x[2])
    df = df.drop(columns=["grids_weights_densities"])

    # Get function evaluations
    df["function_evals1"] = df["grid_points"].apply(
        lambda row: np.array([func1(x) for x in row])
    )
    df["function_evals2"] = df["grid_points"].apply(
        lambda row: np.array([func2(x) for x in row])
    )
    df["function_evals3"] = df["grid_points"].apply(
        lambda row: np.array([func3(x) for x in row])
    )

    # Get covs columns
    df["cov1"] = df.apply(
        lambda row: np.dot(row.weights, row.function_evals1), axis=1
    )
    df["cov2"] = df.apply(
        lambda row: np.dot(row.weights, row.function_evals2), axis=1
    )
    df["cov3"] = df.apply(
        lambda row: np.dot(row.weights, row.function_evals3), axis=1
    )
    df["denom"] = df.apply(lambda row: np.sum(row.weights), axis=1)

    df["cov1"] /= df["denom"]
    df["cov2"] /= df["denom"]
    df["cov3"] /= df["denom"]

    # Get constructed outcomes
    df["outcomes"] = link_function(
        (c1 * df["cov1"] + c2 * df["cov2"] + c3 * df["cov3"])
    )
    if return_dense_df:
        return df
    df_sparse = df.loc[
        :,
        [
            x
            for x in df.columns
            if x
            not in (
                "grid_points",
                "densities",
                "weights",
                "function_evals1",
                "function_evals2",
                "function_evals3",
                "denom",
            )
        ],
    ]
    return (df, df_sparse)
