from data.generate_synthetic_data import additive_noise_a, additive_noise_b, additive_noise_c
from data.generate_synthetic_data import multiplicative_noise_a, multiplicative_noise_b, multiplicative_noise_c
from data.generate_synthetic_data import complex_noise_a, complex_noise_b, complex_noise_c
from data.pairs.generate_pairs import TubingenPairs
import numpy as np


def get_tubingen_pairs_dataset(data_path):
    data_gen = TubingenPairs(path=data_path)

    x, y, weight = [], [], []
    for i in data_gen.pairs_generator():
        x.append(i[0])
        y.append(i[1])
        weight.append(i[2])
    return x, y, weight


def get_synthetic_dataset(
    num_datasets: int,
    sample_size: int,
    func_string: str,
    noise: str
):
    if func_string == "add_a":
        cause, effect = additive_noise_a(
            num_dataset=num_datasets,
            sample_size=sample_size,
            noise=noise
        )
    elif func_string == "add_b":
        cause, effect = additive_noise_b(
            num_dataset=num_datasets,
            sample_size=sample_size,
            noise=noise
        )
    elif func_string == "add_c":
        cause, effect = additive_noise_c(
            num_dataset=num_datasets,
            sample_size=sample_size,
            noise=noise
        )
    elif func_string == "mult_a":
        cause, effect = multiplicative_noise_a(
            num_dataset=num_datasets,
            sample_size=sample_size,
            noise=noise
        )
    elif func_string == "mult_b":
        cause, effect = multiplicative_noise_b(
            num_dataset=num_datasets,
            sample_size=sample_size,
            noise=noise
        )
    elif func_string == "mult_c":
        cause, effect = multiplicative_noise_c(
            num_dataset=num_datasets,
            sample_size=sample_size,
            noise=noise
        )
    elif func_string == "complex_a":
        cause, effect = complex_noise_a(
            num_dataset=num_datasets,
            sample_size=sample_size,
            noise=noise
        )
    elif func_string == "complex_b":
        cause, effect = complex_noise_b(
            num_dataset=num_datasets,
            sample_size=sample_size,
            noise=noise
        )
    elif func_string == "complex_c":
        cause, effect = complex_noise_c(
            num_dataset=num_datasets,
            sample_size=sample_size,
            noise=noise
        )
    else:
        raise NotImplementedError(f"{func_string} has not been implemented!")
    weight = np.ones(num_datasets)
    return cause, effect, weight