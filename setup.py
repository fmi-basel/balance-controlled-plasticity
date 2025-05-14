# setup.py
from setuptools import setup, find_packages

setup(
    name="bcp",         
    version="0.1.0",
    author="Julian Rossbroich",     
    package_dir={"": "bcp"},        
    packages=find_packages(where="bcp"),
    python_requires=">=3.11,<3.12",
)
