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
from eadf.preprocess import *

from . import TestCase


class TestSymmetrizeData(TestCase):
    def setUp(self):
        self.data = np.random.randn(14, 28, 2, 3, 5)

    def test_success(self):
        symmetrizeData(self.data)

    def test_shapeFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "symmetrizeData: got %d dimensions instead of 5" % (4)
        ):
            symmetrizeData(self.data[0])


class TestRegularSamplingToGrid(TestCase):
    def setUp(self):
        self.data = np.random.randn(14 * 13, 2, 3, 5)

    def test_success(self):
        regularSamplingToGrid(self.data, 13, 14)

    def test_shapeFail1(self):
        with self.assertRaisesWithMessage(
            ValueError,
            (
                "regularSamplingToGrid:"
                + "Input arrA has %d dimensions instead of 4"
            )
            % (len(self.data[:, :, :, 0].shape)),
        ):
            regularSamplingToGrid(self.data[:, :, :, 0], 13, 14)

    def test_shapeFail2(self):
        with self.assertRaisesWithMessage(
            ValueError,
            (
                "regularSamplingToGrid:"
                + "numCoEle %d, numAzi %d and arrA.shape[0] %d dont match"
            )
            % (14, 13, self.data[:13].shape[0]),
        ):
            regularSamplingToGrid(self.data[:13], 13, 14)


class TestSampledToFourier(TestCase):
    def setUp(self):
        self.data = np.random.randn(14, 28, 2, 2, 5) + 1j

    def test_success(self):
        sampledToFourier(self.data)


class TestSplineInterpolateFrequency(TestCase):
    def test_success(self):
        splineInterpolateFrequency(
            np.arange(20), np.random.randn(8, 7, 20, 2, 5) + 1j
        )


if __name__ == "__main__":
    unittest.main()
