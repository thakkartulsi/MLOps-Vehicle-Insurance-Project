"""
setup.py

This script is used to package and distribute the Python project using setuptools.

- Defines package metadata such as name, version, author, and email.
- Uses `find_packages()` to automatically detect and include all Python packages.
- Allows the project to be installed as a package via `pip install -e .`.
"""

from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    author="Tulsi Thakkar",
    author_email="thakkartulsi21@gmail.com",
    packages=find_packages()
)