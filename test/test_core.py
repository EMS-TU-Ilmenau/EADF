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


import unittest
import numpy as np

from eadf.backend import *
from eadf.core import *

from . import TestCase


class TestBlockSizeCalculation(TestCase):
    def test_success(self):
        calcBlockSize(
            np.random.randn(5),
            np.random.randn(6),
            np.random.randn(5, 6, 2, 2, 5) + 1j,
            2048,
            True,
        )

        calcBlockSize(
            np.random.randn(5),
            np.random.randn(6),
            np.random.randn(5, 6, 2, 2, 5) + 1j,
            2048,
            False,
        )


class TestPatternTransform(TestCase):
    def testinversePatternTransform_success(self):
        inversePatternTransform(
            np.random.randn(6, 20) + 1j,
            np.random.randn(5, 20) + 1j,
            np.random.randn(6, 5, 2, 2, 5) + 1j,
            7,
        )

    def testinversePatternTransformLowMem_success(self):
        def funCoEle(arrcoEle):
            return 1j * np.outer(np.random.randn(6), arrcoEle)

        def funAzi(arrAzi):
            return 1j * np.outer(np.random.randn(5), arrAzi)

        inversePatternTransformLowMem(
            np.random.randn(20) + 1j,
            np.random.randn(20) + 1j,
            funCoEle,
            funAzi,
            np.random.randn(6, 5, 2, 2, 5) + 1j,
            7,
        )


if __name__ == "__main__":
    unittest.main()
