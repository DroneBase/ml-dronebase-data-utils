#!/usr/bin/env python3

"""Setup script for installing ml-dronebase-data-utils."""

from codecs import open
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

if __name__ == "__main__":
    setup(
        name="ml-dronebase-data-utils",
        version="0.0.6",
        description="A collection of commonly functions used by DroneBase ML Engineers",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/DroneBase/ml-dronebase-data-utils",
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
        packages=["ml_dronebase_data_utils"],
        include_package_data=True,
        install_requires=[
            "boto3>=1.19.2",
            "tqdm>=4.62.3",
            "scikit-learn==1.0.2",
            "Shapely==1.8.1.post1",
            "rasterio==1.2.10",
            "geopandas==0.9.0",
            "Pillow==9.0.0",
            "jinja2>=2.0.1",
            "black>=21.11b1",
            "isort>=5.10.1",
            "colorama",
            "flake8==4.0.1",
            "pytest",
        ],
        entry_points={
            "console_scripts": [
                "convert_geojson = ml_dronebase_data_utils.convert_geojson_cli:convert_geojson_cli",
                "visualize_converted_geojson = ml_dronebase_data_utils.visualize_converted_geojson:visualize_converted_geojson",
            ]
        },
    )
