from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name="DroneBase-ML-Utils",
    version="1.0.0",
    author="Conor Wallace",
    author_email="conor.wallace@dronebase.com",
    description="A collection of commonly used functions for ML devs",
    long_description_content_type="text/markdown",
    url="https://github.com/DroneBase/dronebase-ml-utils",
    install_requires=[
        "boto3==1.19.2"
    ],
    packages=find_namespace_packages(where="dronebase_ml_utils"),
    python_requires=">=3.7",
)