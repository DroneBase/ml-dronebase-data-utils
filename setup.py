from setuptools import setup

setup(
    name="ml-dronebase-utils",
    version="1.0.0",
    author="Conor Wallace",
    author_email="conor.wallace@dronebase.com",
    description="A collection of commonly used functions for ML devs",
    long_description_content_type="text/markdown",
    url="https://github.com/DroneBase/ml-dronebase-utils",
    install_requires=[
        "boto3==1.19.2",
        "tqdm==4.62.3"
    ],
    packages=['ml_dronebase_utils'],
    python_requires=">=3.7",
)
