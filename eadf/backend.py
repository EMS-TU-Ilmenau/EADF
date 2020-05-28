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


r"""
The Backend
-----------

Memory Management
^^^^^^^^^^^^^^^^^

For some calculations it might be a smart idea to process the requested angles
in chunks of a certain size. First, because it might result in fewer cache
misses and second one might be running out of RAM to run the calculations. To
this end, you might set the EADF_LOWMEM environment variable and use it
together with the :py:obj:`.blockSize` property and the
:py:obj:`.optimizeBlockSize` function.

Datatypes
^^^^^^^^^

We support single and double precision calculations. This can be set with the
EADF_DTYPE environment variable. Most of the time single precision allows
approximately twice the computation speed. This is reflected in the
:py:obj:`.dtype` environment variable.

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

A list of all available shell environment variables.

 - EADF_LOWMEM: yay or nay?
 - EADF_DTYPE: single or double?
"""

import logging
import numpy as np
import os

__all__ = [
    "dtype",
    "rDtype",
    "cDtype",
]

_LOWMEM = os.environ.get("EADF_LOWMEM", 0)
_DTYPE = os.environ.get("EADF_DTYPE", "double")

if _LOWMEM:
    lowMemory = True
else:
    lowMemory = False

if _DTYPE == "double":
    dtype = "double"
    rDtype = np.float64
    cDtype = np.complex128
elif _DTYPE == "single":
    dtype = "single"
    rDtype = np.float32
    cDtype = np.complex64
else:
    dtype = "double"
    rDtype = np.float64
    cDtype = np.complex128
    logging.warning("Datatype not implemented. using double.")
