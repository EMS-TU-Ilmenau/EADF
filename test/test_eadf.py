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

import numpy as np
import unittest
from unittest.mock import patch

from eadf.auxiliary import *
from eadf.arrays import *
from eadf.backend import *
from eadf.eadf import *

from . import TestCase


class TestInit(TestCase):
    def setUp(self):
        self.array = generateArbitraryDipole(
            np.random.randn(3, 11),
            np.zeros((3, 11)),
            np.random.randn(1, 11) ** 2,
            np.linspace(4, 4, 1),
        )

        self.data = self.array.pattern(
            *toGrid(self.array.arrCoEle, self.array.arrAzi)
        ).reshape(
            (
                self.array.arrCoEle.shape[0],
                self.array.arrAzi.shape[0],
                1,
                2,
                11,
            )
        )

    def test_arrAziShape(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "EADF:arrAzi.shape%s != expected shape%s"
            % ((self.array.arrAzi.shape[0],), (self.data.shape[1] - 1,)),
        ):
            EADF(
                self.data[:, :-1],
                self.array.arrCoEle,
                self.array.arrAzi,
                self.array.arrFreq,
                self.array.arrPos,
            )

    def test_arrCoEleShape(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "EADF:arrCoEle.shape%s != expected shape%s"
            % ((self.array.arrCoEle.shape[0],), (self.data.shape[0] - 1,)),
        ):
            EADF(
                self.data[:-1],
                self.array.arrCoEle,
                self.array.arrAzi,
                self.array.arrFreq,
                self.array.arrPos,
            )

    def test_arrFreqShape(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "EADF:arrFreq.shape%s != expected shape%s"
            % ((2,), (self.data.shape[2],)),
        ):
            EADF(
                self.data,
                self.array.arrCoEle,
                self.array.arrAzi,
                np.linspace(1, 2, 2),
                self.array.arrPos,
            )

    def test_arrPos1Shape(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "EADF:arrPos.shape%s != expected shape%s"
            % (
                (self.array.arrPos.shape[0] - 1, self.array.arrPos.shape[1]),
                self.array.arrPos.shape,
            ),
        ):
            EADF(
                self.data,
                self.array.arrCoEle,
                self.array.arrAzi,
                self.array.arrFreq,
                self.array.arrPos[:-1],
            )

    def test_arrPos2Shape(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "EADF:arrPos.shape%s != expected shape%s"
            % (
                (self.array.arrPos.shape[0], self.array.arrPos.shape[1] - 1),
                self.array.arrPos.shape,
            ),
        ):
            EADF(
                self.data,
                self.array.arrCoEle,
                self.array.arrAzi,
                self.array.arrFreq,
                self.array.arrPos[:, :-1],
            )


class TestProperties(TestCase):
    def setUp(self):
        self.array = generateArbitraryDipole(
            np.random.randn(3, 11),
            np.zeros((3, 11)),
            np.random.randn(1, 11) ** 2,
            np.linspace(4, 5, 3),
        )

    def test_arrComFactSuccess(self):
        self.array.compressionFactor = 0.9
        self.assertTrue(self.array.compressionFactor >= 0.9)

    def test_arrComFact1(self):
        with self.assertRaisesWithMessage(
            ValueError, "Supplied Value must be in (0, 1]"
        ):
            self.array.compressionFactor = 1.1

    def test_arrComFact2(self):
        with self.assertRaisesWithMessage(
            ValueError, "Supplied Value must be in (0, 1]"
        ):
            self.array.compressionFactor = -1.1

    def test_operFreq_success(self):
        self.array.operatingFrequencies = np.linspace(4, 5, 5)

    def test_operFreq_singlefreqfail(self):
        with self.assertRaisesWithMessage(
            ValueError, "Not possible in the single freq case"
        ):
            array = generateArbitraryDipole(
                np.random.randn(3, 11),
                np.zeros((3, 11)),
                np.random.randn(1, 11) ** 2,
                np.linspace(5, 5, 1),
            )
            array.operatingFrequencies = np.linspace(4, 5, 5)

    def test_operFreq_lowfreqfail(self):
        with self.assertRaisesWithMessage(
            ValueError, "Unallowed Frequency, too small"
        ):
            self.array.operatingFrequencies = np.linspace(3, 5, 10)

    def test_operFreq_highfreqfail(self):
        with self.assertRaisesWithMessage(
            ValueError, "Unallowed Frequency, too large"
        ):
            self.array.operatingFrequencies = np.linspace(4, 5.5, 10)

    def test_version_success(self):
        from eadf import __version__

        self.assertTrue(self.array.version == __version__)


class TestSubbands(TestCase):
    def setUp(self):
        self.array = generateArbitraryDipole(
            np.random.randn(3, 11),
            np.zeros((3, 11)),
            np.random.randn(1, 11) ** 2,
            np.linspace(2, 4, 15),
        )

    def test_success(self):
        self.array.defineSubBands(
            np.linspace(2, 4, 3), self.array.arrFourierData[:, :, :3]
        )

    def test_shape_fail(self):
        with self.assertRaisesWithMessage(
            ValueError, "intervals.shape[0] != data.shape[2]"
        ):
            self.array.defineSubBands(
                np.linspace(2, 4, 3), self.array.arrFourierData[:, :, :4]
            )

    def test_singlefreq_fail(self):
        with self.assertRaisesWithMessage(
            ValueError, "Not possible in the single freq case"
        ):
            array = generateArbitraryDipole(
                np.random.randn(3, 11),
                np.zeros((3, 11)),
                np.random.randn(1, 11) ** 2,
                np.linspace(4, 4, 1),
            )
            array.defineSubBands(
                np.linspace(4, 4, 1), array.arrFourierData[:, :, :1]
            )

    def test_lowerend_fail(self):
        with self.assertRaisesWithMessage(
            ValueError, "Intervals must start at self.arrFreq[0]"
        ):
            self.array.defineSubBands(
                np.linspace(2.1, 4, 3), self.array.arrFourierData[:, :, :3]
            )

    def test_upperend_fail(self):
        with self.assertRaisesWithMessage(
            ValueError, "Intervals must end at self.arrFreq[-1]"
        ):
            self.array.defineSubBands(
                np.linspace(2, 4.1, 3), self.array.arrFourierData[:, :, :3]
            )

    def test_sorted_fail(self):
        with self.assertRaisesWithMessage(
            ValueError, "Interval bounds must be sorted"
        ):
            self.array.defineSubBands(
                np.linspace(2, 4, 4)[[0, 2, 1, 3]],
                self.array.arrFourierData[:, :, :4],
            )

    def test_assignmentArray_success(self):
        self.array.defineSubBands(
            np.linspace(2, 4, 4), self.array.arrFourierData[:, :, :4]
        )
        # self.array.


class TestPattern(TestCase):
    def setUp(self):
        self.array = generateArbitraryDipole(
            np.random.randn(3, 11),
            np.zeros((3, 11)),
            np.random.randn(1, 11) ** 2,
            np.linspace(4, 4, 1),
        )
        self.arrCoEle, self.arrAzi = toGrid(*sampleAngles(5, 10))

    def test_success_pattern(self):
        self.array.pattern(self.arrCoEle, self.arrAzi)

    def test_success_gradient(self):
        self.array.gradient(self.arrCoEle, self.arrAzi)

    def test_success_hessian(self):
        self.array.hessian(self.arrCoEle, self.arrAzi)

    def test_inputSizeFail(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "eval: supplied angle arrays have size %d and %d."
            % (self.arrCoEle[:-1].shape[0], self.arrAzi.shape[0]),
        ):
            self.array.pattern(self.arrCoEle[:-1], self.arrAzi)


class TestSerialization(unittest.TestCase):
    def setUp(self):
        self.array = generateArbitraryDipole(
            np.random.randn(3, 11),
            np.zeros((3, 11)),
            np.random.randn(1, 11) ** 2,
            np.linspace(4, 4, 1),
        )

    def test_save_success(self):
        self.array.save("test.dat")

    def test_load_success(self):
        self.array.save("test.dat")
        arrayLoad = EADF.load("test.dat")

    @patch("logging.warning")
    def test_load_version_warn(self, mock):
        self.array._version += "bla"
        self.array.save("test.dat")
        arrayLoad = EADF.load("test.dat")
        mock.assert_called_with(
            "eadf.load: loaded object does not match current version."
        )


if __name__ == "__main__":
    unittest.main()
