#!/usr/bin/env python

import sys
# To use a consistent encoding
from codecs import open
from os import path

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from sphinx.setup_command import BuildDoc

import versioneer


### py.test ###
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['marxs', 'docs']

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


cmdclass = versioneer.get_cmdclass()
cmdclass['test'] = PyTest
cmdclass['build_sphinx'] = BuildDoc

setup_args = {
    'name': 'filili',
    'description': 'Fit list of spectral lines to data using sherpa',
    'author': 'MIT / H. M. Guenther',
    'author_email': 'hgunther@mit.edu',

    'version': versioneer.get_version(),
    'cmdclass': cmdclass,

    'license': 'MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    'classifiers': [
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 2 - Pre-Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy'
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure.
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        ],

    'packages': find_packages(),
    'package_data' = {
        'sample_linelists': ['*.dat'],
        },

    'setup_requires': ["sherpa"],
    'install_requires': [],
    'tests_require': ['pytest'],
    }

### import README file ###
here = path.abspath(path.dirname(__file__))
# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    setup_args['long_description'] = f.read()


setup(**setup_args)
