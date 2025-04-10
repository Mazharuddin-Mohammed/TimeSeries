from setuptools import setup, find_packages

setup(
    name="tsx",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pybind11>=2.6"],
    package_data={"tsx": ["tsx_backend*.so"]},
    author="Your Name",
    description="Time Series Analysis Library",
    license="MIT",
)