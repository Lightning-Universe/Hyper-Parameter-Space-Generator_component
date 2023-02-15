#!/usr/bin/env python

import os

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)


def _load_requirements(path_dir: str = _PATH_ROOT, file_name: str = "requirements.txt") -> list:
    reqs = parse_requirements(open(os.path.join(path_dir, file_name)).readlines())
    return list(map(str, reqs))


setup(
    name="hp_space_generator",
    version="0.1.0",
    description="Generator Hyper Parameter space around the given choices of hyper-parameters with Grid Search and Random Search strategies",  # noqa: E501
    author="Kushashwa Ravi Shrimali, Ethan Harris",
    author_email="kush@lightning.ai",
    url="https://github.com/Lightning-AI/LAI-Hyper-Parameter-Space-Generator-Component",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=_load_requirements(),
)
