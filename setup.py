#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

import sys
import warnings
import versioneer

from setuptools import setup, find_packages

setup(
    name='pySMI',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Brookhaven National Laboratory_SMI',
    packages=find_packages(),
    #package_data={'chxtools/X-ray_database' : ['.dat']},    include_package_data=True,
)
