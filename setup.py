#!/usr/bin/env python3

"""Setup script for installing ml-dronebase-utils."""

from codecs import open
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

if __name__ == "__main__":
    setup(
        name="ml-dronebase-utils",
        version="0.5",
        description="A collection of commonly functions used by DroneBase ML Engineers",
        long_description=long_description,
        url="https://github.com/DroneBase/ml-dronebase-utils",
        author="Conor Wallace",
        author_email="conor.wallace@dronebase.com",
        license="MIT",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        keywords="python, utilities",
        packages=["ml_dronebase_utils"],
        include_package_data=True,
        install_requires=[
            "boto3>=1.19.2",
            "tqdm>=4.62.3",
            "jinja2>=2.0.1",
            "black",
            "isort",
            "colorama",
            "flake8",
            "pytest",
        ],
    )
