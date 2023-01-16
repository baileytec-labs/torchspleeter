from __future__ import absolute_import
from __future__ import print_function
from setuptools import setup, find_packages
import distutils.text_file
from pathlib import Path
from typing import List
#import os
#dirname = os.path.dirname(__file__)
#requirementsfile = os.path.join(dirname, 'REQUIREMENTS.txt')
#requirementsfile="requirements.txt"
# Always prefer setuptools over distutils
import setuptools

def _parse_requirements(filename: str) -> List[str]:
    """Return requirements from requirements file."""
    # Ref: https://stackoverflow.com/a/42033122/
    return distutils.text_file.TextFile(filename=str(Path(__file__).with_name(filename))).readlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

print(find_packages())

setup(
    name='torchspleeter',
    version='0.0.9',
    packages=find_packages(),
    entry_points={
        "console_scripts": ["torchspleeter=torchspleeter.command_interface:main"]
    },
    author="Sean Bailey",
    author_email="seanbailey518@gmail.com",
    description="This provides a port of spleeter in Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sean-bailey",
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
install_requires=[
    "torch",
    "numpy",
    "librosa",
    "pydub"
]

)
#install_requires=_parse_requirements(requirementsfile),