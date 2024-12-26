from integral_functions.config import AGE_MID
from integral_functions.typing import Float_or_array, Numeric


def f(x: Float_or_array) -> Float_or_array:
    return -(((1 / 5) * x - 5) ** 2) + 40


def g(x: Float_or_array) -> Float_or_array:
    return ((x / 15) - 8.3335) ** 2


def age_distribution(age: Numeric):
    if 0 <= age <= AGE_MID:
        return f(age)
    else:
        return g(age)
