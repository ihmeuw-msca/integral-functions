from typing import Callable

from scipy.integrate import quad


def integrate_expectation(
    func: Callable,
    density: Callable,
    age_start: float,
    age_end: float,
    ret_error: bool = False,
) -> float:
    def function_multiplier(x: float):
        return func(x) * density(x)

    integral_value, error_approx = quad(function_multiplier, age_start, age_end)
    if ret_error:
        return integral_value, error_approx
    return integral_value


def integrate_density(
    density: Callable,
    age_start: float,
    age_end: float,
    ret_error: bool = False,
) -> float:
    integral_value, error_approx = quad(density, age_start, age_end)
    if ret_error:
        return integral_value, error_approx
    return integral_value
