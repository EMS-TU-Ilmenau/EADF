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

from eadf.auxiliary import *
from eadf.backend import *

from . import TestCase


class TestCartesianToSpherical(TestCase):
    def setUp(self):
        self.arrA = np.random.randn(10, 3)

    def test_success(self):
        cartesianToSpherical(self.arrA)

    def test_inputSizeFail(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "cartesianToSpherical: arrA has wrong second dimension.",
        ):
            cartesianToSpherical(self.arrA[:, :2])

    def test_inputTypeFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "cartesianToSpherical: arrA is complex."
        ):
            cartesianToSpherical(self.arrA + 1j)


class TestColumnwiseKron(TestCase):
    def setUp(self):
        self.arrA = np.random.randn(5, 3)
        self.arrB = np.random.randn(7, 3)

    def test_success(self):
        columnwiseKron(self.arrA, self.arrB)

    def test_inputSizeFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "columnwiseKron: Matrices cannot be multiplied"
        ):
            columnwiseKron(self.arrA[:, :2], self.arrB)


class TestSampleAngles(TestCase):
    def test_sucess1(self):
        sampleAngles(10, 15)

    def test_sucess2(self):
        sampleAngles(10, 15, lstEndPoints=[False, False])

    def test_sucess3(self):
        sampleAngles(10, 15, lstEndPoints=[True, False])

    def test_sucess4(self):
        sampleAngles(10, 15, lstEndPoints=[False, True])

    def test_sucess5(self):
        sampleAngles(10, 15, lstEndPoints=[True, True])

    def test_inputFail1(self):
        with self.assertRaisesWithMessage(
            ValueError, "sampleAngles: numAzi is %d, must be > 0" % (-1)
        ):
            sampleAngles(5, -1)

    def test_inputFail2(self):
        with self.assertRaisesWithMessage(
            ValueError, "sampleAngles: numCoEle is %d, must be > 0" % (-1)
        ):
            sampleAngles(-1, 10)

    def test_inputFail3(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "sampleAngles: lstEndPoints has length %d instead of 2." % (3),
        ):
            sampleAngles(10, 10, lstEndPoints=[False, False, False])


class TestToGrid(TestCase):
    def test_success1(self):
        toGrid(np.arange(4), np.arange(5))

    def test_success2(self):
        toGrid(np.arange(4), np.arange(5), np.arange(6))


if __name__ == "__main__":
    unittest.main()
