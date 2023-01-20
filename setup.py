from setuptools import setup, find_packages

pkgs = find_packages()
setup(
    name="gplvm_causal_discovery",
    version="0.0.1",
    packages=pkgs,
    long_description=open("README.md").read(),
)
