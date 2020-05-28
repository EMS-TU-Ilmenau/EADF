# Copyright 2020 EMS Group TU Ilmenau
#     https://www.tu-ilmenau.de/it-ems/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Usage:
# Docu: python setup.py build_sphinx -E
# Test: python setup.py test

from setuptools import setup
from eadf import __version__
from sphinx.setup_command import BuildDoc
from os import path

cmdclass = {"build_sphinx": BuildDoc}

name = "eadf"
author = "EMS Group, TU Ilmenau, 2020"
version = __version__
release = ".".join(version.split(".")[:2])

# read the contents of the README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    author=author,
    version=version,
    name=name,
    packages=[name],
    author_email="sebastian.semper@tu-ilmenau.de",
    description="Effective Aperture Distribution Function",
    url="https://eadf.readthedocs.io/en/latest/",
    license="Apache Software License",
    keywords="signal processing, array processing",
    test_suite="test",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    command_options={
        "build_sphinx": {
            "project": ("setup.py", name),
            "copyright": ("setup.py", author),
            "version": ("setup.py", version),
            "release": ("setup.py", release),
            "source_dir": ("setup.py", "doc/source"),
            "build_dir": ("setup.py", "doc/build"),
        }
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "sphinx",
        "sphinx_rtd_theme",
        "matplotlib",
        "tornado",
        "numpy",
        "scipy",
        "six",
        "pycodestyle",
        "codecov",
        "pandas",
        "pint==0.9",
    ],
)
