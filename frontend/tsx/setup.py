"""
@file setup.py
@brief Setup script for the tsx Python package.

This file contains the setup configuration for the tsx Python package,
which provides a Python interface to the C++ TimeSeries library.
It handles package metadata, dependencies, and installation.

@author Mazharuddin Mohammed
"""

from setuptools import setup, find_packages

setup(
    name="tsx",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pybind11>=2.6"],
    package_data={"tsx": ["tsx_backend*.so"]},
    author="Mazhar",
    description="Time Series Analysis Library",
    license="MIT",
)