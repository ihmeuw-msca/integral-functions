from typing import Callable

from scipy.integrate import quad


def integrate_cov(
    func: Callable,
    density: Callable,
    age_start: float,
    age_end: float,
    age_mid: int,
    ret_error: bool = False,
) -> float:
    def function_multiplier(x: float):
        return func(x) * density(x, age_mid)

    if age_start < age_mid < age_end:
        integral_value1, error_approx1 = quad(
            function_multiplier, age_start, age_mid
        )
        integral_value2, error_approx2 = quad(
            function_multiplier, age_mid, age_end
        )
        integral_value = integral_value1 + integral_value2
        error_approx = error_approx1 + error_approx2
    else:
        integral_value, error_approx = quad(
            function_multiplier, age_start, age_end
        )
    if ret_error:
        return integral_value, error_approx
    return integral_value


def integrate_denom(
    density: Callable,
    age_start: float,
    age_end: float,
    age_mid: int,
    ret_error: bool = False,
) -> float:
    if age_start < age_mid < age_end:
        integral_value1, error_approx1 = quad(
            density, age_start, age_mid, args=age_mid
        )
        integral_value2, error_approx2 = quad(
            density, age_mid, age_end, args=age_mid
        )
        integral_value = integral_value1 + integral_value2
        error_approx = error_approx1 + error_approx2
    else:
        integral_value, error_approx = quad(
            density, age_start, age_end, args=age_mid
        )
    if ret_error:
        return integral_value, error_approx
    return integral_value


def integrate_expecatation(
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
