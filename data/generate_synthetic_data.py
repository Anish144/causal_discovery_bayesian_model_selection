import numpy as np


def normal_noise(
    num_dataset: int,
    sample_size: int
):
    return np.random.normal(loc=0., scale=1., size=(num_dataset, sample_size, 1))


def uniform_noise(
    num_dataset: int,
    sample_size: int
):
    return np.random.uniform(low=0., high=1., size=(num_dataset, sample_size, 1))


def exp_noise(
    num_dataset: int,
    sample_size: int
):
    return np.random.exponential(scale=1., size=(num_dataset, sample_size, 1))


def additive_noise_a(
    num_dataset: int,
    sample_size: int,
    noise: str
):
    cause = np.random.normal(
        loc=0., scale=1., size=(num_dataset, sample_size, 1)
    )
    effect = cause ** 3 + cause + eval(f"{noise}_noise")(num_dataset, sample_size)
    return cause, effect


def additive_noise_b(
    num_dataset: int,
    sample_size: int,
    noise: str
):
    cause = np.random.normal(
        loc=0., scale=1., size=(num_dataset, sample_size, 1)
    )
    effect = np.log(cause + 10) + cause ** 6 + eval(f"{noise}_noise")(num_dataset, sample_size)
    return cause, effect


def additive_noise_c(
    num_dataset: int,
    sample_size: int,
    noise: str
):
    cause = np.random.normal(
        loc=0., scale=1., size=(num_dataset, sample_size, 1)
    )
    effect = np.sin(10 * cause) + np.exp(3 * cause) + eval(f"{noise}_noise")(num_dataset, sample_size)
    return cause, effect


def multiplicative_noise_a(
    num_dataset: int,
    sample_size: int,
    noise: str
):
    cause = np.random.normal(
        loc=0., scale=1., size=(num_dataset, sample_size, 1)
    )
    effect = (cause ** 3 + cause) * np.exp(eval(f"{noise}_noise")(num_dataset, sample_size))
    return cause, effect


def multiplicative_noise_b(
    num_dataset: int,
    sample_size: int,
    noise: str
):
    cause = np.random.normal(
        loc=0., scale=1., size=(num_dataset, sample_size, 1)
    )
    effect = (np.sin(10 * cause) + np.exp(3 * cause)) * np.exp(eval(f"{noise}_noise")(num_dataset, sample_size))
    return cause, effect


def multiplicative_noise_c(
    num_dataset: int,
    sample_size: int,
    noise: str
):
    cause = np.random.normal(
        loc=0., scale=1., size=(num_dataset, sample_size, 1)
    )
    effect = (np.log(10 + cause) + cause ** 6) * np.exp(eval(f"{noise}_noise")(num_dataset, sample_size))
    return cause, effect


def complex_noise_a(
    num_dataset: int,
    sample_size: int,
    noise: str
):
    cause = np.random.normal(
        loc=0., scale=1., size=(num_dataset, sample_size, 1)
    )
    effect = np.power(np.log(10 + cause) + cause ** 2, eval(f"{noise}_noise")(num_dataset, sample_size))
    return cause, effect


def complex_noise_b(
    num_dataset: int,
    sample_size: int,
    noise: str
):
    cause = np.random.normal(
        loc=0., scale=1., size=(num_dataset, sample_size, 1)
    )
    noise_term = np.abs(eval(f"{noise}_noise")(num_dataset, sample_size))
    effect = np.log(cause + 10) + np.power(cause ** 2, noise_term)
    return cause, effect


def complex_noise_c(
    num_dataset: int,
    sample_size: int,
    noise: str
):
    cause = np.random.normal(
        loc=0., scale=1., size=(num_dataset, sample_size, 1)
    )
    effect = np.log(cause ** 6 + 5) + cause ** 5 - np.sin((cause ** 2) * np.abs(eval(f"{noise}_noise")(num_dataset, sample_size)))
    return cause, effect