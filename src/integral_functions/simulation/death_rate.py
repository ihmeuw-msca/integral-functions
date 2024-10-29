from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from integral_functions.typing import Numeric


def func1(x: NDArray) -> NDArray:
    return x


def func2(x: NDArray) -> NDArray:
    return -np.sqrt(x)


def func3(x: NDArray) -> NDArray:
    return x**2


def death_rate_function(
    age: NDArray,
    c1: Numeric = 0.1,
    c2: Numeric = 1,
    c3: Numeric = 0.0017,
    func1: Callable = func1,
    func2: Callable = func2,
    func3: Callable = func3,
) -> NDArray:
    return c1 * func1(age) + c2 * func2(age) + c3 * func3(age)


def expit_death_rate_function(
    age: NDArray,
    c1: Numeric = 0.1,
    c2: Numeric = 1,
    c3: Numeric = 0.0017,
    func1: Callable = func1,
    func2: Callable = func2,
    func3: Callable = func3,
):
    return expit(
        death_rate_function(
            age, c1=c1, c2=c2, c3=c3, func1=func1, func2=func2, func3=func3
        )
    )
