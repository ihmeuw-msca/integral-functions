from integral_functions.typing import Float_or_array, Numeric


def f(x: Float_or_array) -> Float_or_array:
    return -(((1 / 5) * x - 5) ** 2) + 40


def g(x: Float_or_array) -> Float_or_array:
    return ((x / 15) - 8.3335) ** 2


def age_distribution(age: Numeric, age_mid: Numeric):
    if 0 <= age <= age_mid:
        return f(age)
    else:
        return g(age)
