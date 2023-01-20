from .generate_synthetic_data import (
    additive_noise_a,
    additive_noise_b,
    additive_noise_c,
)
from .generate_synthetic_data import (
    multiplicative_noise_a,
    multiplicative_noise_b,
    multiplicative_noise_c,
)
from .generate_synthetic_data import (
    complex_noise_a,
    complex_noise_b,
    complex_noise_c,
)
from .cha_pairs.generate_cha_pairs import ChaPairs
from .gauss_pairs.generate_gauss_pairs import GaussPairs
from .multi_pairs.generate_multi_pairs import MultiPairs
from .net_pairs.generate_net_pairs import NetPairs
from .pairs.generate_pairs import TubingenPairs
from .gplvm_pairs.generate_gplvm_pairs import GPLVMPairs
from .linear_pairs.generate_linear_pairs import LinearPairs
import numpy as np


def get_linear_pairs_dataset(data_path):
    data_gen = LinearPairs(path=data_path)

    x, y, weight, target = data_gen.return_pairs()
    return x, y, weight, target


def get_gplvm_pairs_dataset(data_path):
    data_gen = GPLVMPairs(path=data_path)

    x, y, weight, target = data_gen.return_pairs()
    return x, y, weight, target


def get_net_pairs_dataset(data_path):
    data_gen = NetPairs(path=data_path)

    x, y, weight, target = data_gen.return_pairs()
    return x, y, weight, target


def get_multi_pairs_dataset(data_path):
    data_gen = MultiPairs(path=data_path)

    x, y, weight, target = data_gen.return_pairs()
    return x, y, weight, target


def get_gauss_pairs_dataset(data_path):
    data_gen = GaussPairs(path=data_path)

    x, y, weight, target = data_gen.return_pairs()
    return x, y, weight, target


def get_cha_pairs_dataset(data_path):
    data_gen = ChaPairs(path=data_path)

    x, y, weight, target = data_gen.return_pairs()
    return x, y, weight, target


def get_tubingen_pairs_dataset(data_path):
    data_gen = TubingenPairs(path=data_path)
    x, y, weight = [], [], []
    for i in data_gen.pairs_generator():
        x.append(i[0])
        y.append(i[1])
        weight.append(i[2])
    target = np.ones(len(x), dtype=np.float64)
    return x, y, weight, target


def get_synthetic_dataset(
    num_datasets: int, sample_size: int, func_string: str, noise: str
):
    if func_string == "add_a":
        cause, effect = additive_noise_a(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "add_b":
        cause, effect = additive_noise_b(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "add_c":
        cause, effect = additive_noise_c(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "mult_a":
        cause, effect = multiplicative_noise_a(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "mult_b":
        cause, effect = multiplicative_noise_b(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "mult_c":
        cause, effect = multiplicative_noise_c(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "complex_a":
        cause, effect = complex_noise_a(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "complex_b":
        cause, effect = complex_noise_b(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "complex_c":
        cause, effect = complex_noise_c(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    else:
        raise NotImplementedError(f"{func_string} has not been implemented!")
    weight = np.ones(num_datasets)
    return cause, effect, weight
