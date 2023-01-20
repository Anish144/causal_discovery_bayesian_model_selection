from setuptools import setup, find_packages

pkgs = find_packages()
req = [
    "dill==0.3.5.1",
    "gpflow==2.5.2",
    "matplotlib==3.5.2",
    "numpy==1.22.4",
    "pandas==1.5.0",
    "scikit_learn==1.2.0",
    "setuptools==44.0.0",
    "tensorflow==2.8.2",
    "tensorflow_probability==0.16.0",
    "tqdm==4.64.0",
]
setup(
    name="gplvm_causal_discovery",
    version="0.0.1",
    packages=pkgs,
    long_description=open("README.md").read(),
    install_requires=req,
)
