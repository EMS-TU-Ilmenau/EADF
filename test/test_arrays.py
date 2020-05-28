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
import os

from eadf.arrays import *
from eadf.auxiliary import *
from eadf.backend import *
from eadf.importers import *

from . import TestCase


class TestURA(TestCase):
    def setUp(self):
        self.array = generateURA(5, 6, 0.5, 0.75, np.ones(1), np.ones(1))

    def test_numElements(self):
        self.assertEqual(self.array.numElements, 5 * 6)

    def test_arrPosSize(self):
        self.assertEqual(self.array.arrPos.shape, (3, 5 * 6))

    def test_numElementsXFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateURA: numElementsY <= 0 is not allowed."
        ):
            # must not be negative
            self.array = generateURA(-5, 6, 0.5, 0.5, np.ones(1), np.ones(1))

    def test_numSpacingXFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateURA: numSpacingY <= 0 is not allowed."
        ):
            self.array = generateURA(5, 6, -0.5, 0.5, np.ones(1), np.ones(1))

    def test_numElementsYFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateURA: numElementsZ <= 0 is not allowed."
        ):
            self.array = generateURA(5, -6, 0.5, 0.5, np.ones(1), np.ones(1))

    def test_numSpacingYFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateURA: numSpacingZ <= 0 is not allowed."
        ):
            self.array = generateURA(5, 6, 0.5, -0.5, np.ones(1), np.ones(1))

    def test_elementSizeFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateURA: elementSize <= 0 is not allowed."
        ):
            self.array = generateURA(5, 6, 0.5, 0.5, -np.ones(1), np.ones(1))

    def test_elementSizeFail2(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "generateURA: elementSize has to be a 1- or 3-element array.",
        ):
            self.array = generateURA(
                5, 6, 0.5, 0.5, np.ones((2, 1)), np.ones(1),
            )

    def test_frequencyFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateURA: frequency <= 0 is not allowed."
        ):
            self.array = generateURA(5, 6, 0.5, 0.5, np.ones(1), -np.ones(1))


class TestULA(TestCase):
    def setUp(self):
        self.array = generateULA(11, 0.5, np.ones(1), np.ones(1))

    def test_numElements(self):
        self.assertEqual(self.array.numElements, 11)

    def test_arrPosSize(self):
        self.assertEqual(self.array.arrPos.shape, (3, 11))

    def test_numElementsFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateULA: numElements <= 0 is not allowed."
        ):
            self.array = generateULA(-11, 0.5, np.ones(1), np.ones(1))

    def test_numSpacingFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateULA: numSpacing <= 0 is not allowed."
        ):
            self.array = generateULA(11, -0.5, np.ones(1), np.ones(1))

    def test_elementSizeFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateULA: elementSize <= 0 is not allowed."
        ):
            self.array = generateULA(11, 0.5, -np.ones(1), np.ones(1))

    def test_elementSizeFail2(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "generateULA: elementSize has to be a 1- or 3-element array.",
        ):
            self.array = generateULA(11, 0.5, np.ones((2, 1)), np.ones(1))

    def test_frequencyFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateULA: frequency <= 0 is not allowed."
        ):
            self.array = generateULA(11, 0.5, np.ones(1), -np.ones(1))


class TestUCA(TestCase):
    def setUp(self):
        self.array = generateUCA(11, 0.5, np.ones(1), np.ones(1))

    def test_numElements(self):
        self.assertEqual(self.array.numElements, 11)

    def test_arrPosSize(self):
        self.assertEqual(self.array.arrPos.shape, (3, 11))

    def test_numElementsFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateUCA: numElements <= 0 is not allowed."
        ):
            self.array = generateUCA(-11, 0.5, np.ones(1), np.ones(1))

    def test_numRadiusFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateUCA: numRadius <= 0 is not allowed."
        ):
            self.array = generateUCA(11, -0.5, np.ones(1), np.ones(1))

    def test_elementSizeFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateUCA: elementSize <= 0 is not allowed."
        ):
            self.array = generateUCA(11, 0.5, -np.ones(1), np.ones(1))

    def test_elementSizeFail2(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "generateUCA: elementSize has to be a 1- or 3-element array.",
        ):
            self.array = generateUCA(11, 0.5, np.ones((2, 1)), np.ones(1))

    def test_frequencyFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateUCA: frequency <= 0 is not allowed."
        ):
            self.array = generateUCA(11, 0.5, np.ones(1), -np.ones(1))


class TestStackedUCA(TestCase):
    def setUp(self):
        self.array = generateStackedUCA(
            11, 3, 0.5, 0.5, np.ones(1), np.ones(1)
        )

    def test_numElements(self):
        self.assertEqual(self.array.numElements, 11 * 3)

    def test_arrPosSize(self):
        self.assertEqual(self.array.arrPos.shape, (3, 11 * 3))

    def test_numElementsFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateStackedUCA: numElements <= 0 is not allowed."
        ):
            self.array = generateStackedUCA(
                -11, 3, 0.5, 0.5, np.ones(1), np.ones(1)
            )

    def test_numStacksFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateStackedUCA: numStacks <= 0 is not allowed."
        ):
            self.array = generateStackedUCA(
                11, -3, 0.5, 0.5, np.ones(1), np.ones(1)
            )

    def test_numRadiusFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateStackedUCA: numRadius <= 0 is not allowed."
        ):
            self.array = generateStackedUCA(
                11, 3, -0.5, 0.5, np.ones(1), np.ones(1)
            )

    def test_numHeightFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateStackedUCA: numHeight <= 0 is not allowed."
        ):
            self.array = generateStackedUCA(
                11, 3, 0.5, -0.5, np.ones(1), np.ones(1)
            )

    def test_elementSizeFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateStackedUCA: elementSize <= 0 is not allowed."
        ):
            self.array = generateStackedUCA(
                11, 3, 0.5, 0.5, -np.ones(1), np.ones(1)
            )

    def test_elementSizeFail2(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "generateURA: elementSize has to be a 1- or 3-element array.",
        ):
            self.array = generateURA(
                11, 3, 0.5, 0.5, np.ones((2, 1)), np.ones(1),
            )

    def test_frequencyFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateStackedUCA: frequency <= 0 is not allowed."
        ):
            self.array = generateStackedUCA(
                11, 3, 0.5, 0.5, np.ones(1), -np.ones(1)
            )


class TestArbitraryDipole(TestCase):
    def setUp(self):
        self.arrFreq = np.array([5e9, 5.2e9])
        self.arrPos = 3e8 / np.mean(self.arrFreq) / 2 * np.random.randn(3, 4)
        self.arrRot = np.zeros((3, 4))
        self.arrLength = (
            3e8 / np.mean(self.arrFreq) / 2 * np.abs(np.random.randn(1, 4))
        )
        self.array = generateArbitraryDipole(
            self.arrPos, self.arrRot, self.arrLength, self.arrFreq,
        )
        self.arrayDualPol = generateArbitraryDipole(
            self.arrPos,
            self.arrRot,
            self.arrLength,
            self.arrFreq,
            addCrossPolPort=True,
        )

    def test_numElements(self):
        self.assertEqual(self.array.numElements, 4)

    def test_numElementsDualPol(self):
        self.assertEqual(self.arrayDualPol.numElements, 2 * 4)

    def test_arrPosSize(self):
        self.assertEqual(self.array.arrPos.shape, (3, 4))

    def test_arrPosFail(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "generateArbitraryDipole: arrPos must have exactly 3 rows",
        ):
            generateArbitraryDipole(
                np.random.randn(2, 4),
                self.arrRot,
                self.arrLength,
                self.arrFreq,
            )

    def test_arrRotFail(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "generateArbitraryDipole: arrRot must have exactly 3 rows",
        ):
            generateArbitraryDipole(
                self.arrPos, self.arrRot.T, self.arrLength, self.arrFreq,
            )

    def test_arrLengthFail(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "generateArbitraryDipole: arrLength must be a rowvector (1 x N)",
        ):
            generateArbitraryDipole(
                self.arrPos, self.arrRot, self.arrLength.T, self.arrFreq,
            )

    def test_rotXFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateArbitraryDipole: only rotation around z-axis",
        ):
            arrRot = self.arrRot
            arrRot[0, 0] = 1
            generateArbitraryDipole(
                self.arrPos, arrRot, self.arrLength, self.arrFreq,
            )

    def test_rotYFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateArbitraryDipole: only rotation around z-axis",
        ):
            arrRot = self.arrRot
            arrRot[1, 0] = 1
            generateArbitraryDipole(
                self.arrPos, arrRot, self.arrLength, self.arrFreq,
            )


class TestArbitraryPatch(TestCase):
    def setUp(self):
        self.arrPos = np.stack(
            (
                np.array([0, 0, 0]),
                np.array([0, 0, 35e-3]),
                np.array([0, 35e-3, 0]),
                np.array([0, 35e-3, 35e-3]),
            ),
            axis=0,
        ).T
        self.arrRot = np.zeros((3, 4))
        self.arrSize = np.repeat(np.array([[30e-3, 30e-3, 1e-3]]).T, 4, axis=1)
        self.arrFreq1 = np.array([5e9])
        self.array = generateArbitraryPatch(
            self.arrPos, self.arrRot, self.arrSize, self.arrFreq1,
        )
        self.arrayDualPol = generateArbitraryPatch(
            self.arrPos,
            self.arrRot,
            self.arrSize,
            self.arrFreq1,
            addCrossPolPort=True,
        )
        # path to the Matlab EADF for comparison of the values
        dirname = os.path.dirname(__file__)
        self.matlabEADFPath = os.path.join(
            dirname, "data/NarrowBandPatchArray.mat"
        )

    def test_numElements(self):
        self.assertEqual(self.array.numElements, 4)

    def test_numElementsDualPol(self):
        self.assertEqual(self.arrayDualPol.numElements, 2 * 4)

    def test_arrPosSize(self):
        self.assertEqual(self.array.arrPos.shape, (3, 4))

    def test_arrPosFail(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "generateArbitraryPatch: arrPos must have exactly 3 rows",
        ):
            generateArbitraryPatch(
                self.arrPos.T, self.arrRot, self.arrSize, self.arrFreq1,
            )

    def test_arrRotFail(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "generateArbitraryPatch: arrRot must have exactly 3 rows",
        ):
            generateArbitraryPatch(
                self.arrPos, self.arrRot.T, self.arrSize, self.arrFreq1,
            )

    def test_arrSizeFail(self):
        with self.assertRaisesWithMessage(
            ValueError,
            "generateArbitraryPatch: arrSize must have exactly 3 rows",
        ):
            generateArbitraryPatch(
                self.arrPos, self.arrRot, self.arrSize.T, self.arrFreq1,
            )

    def test_rotXFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateArbitraryPatch: only rotation around z-axis",
        ):
            arrRot = self.arrRot
            arrRot[0, 0] = 1
            generateArbitraryPatch(
                self.arrPos, arrRot, self.arrSize, self.arrFreq1,
            )

    def test_rotYFail(self):
        with self.assertRaisesWithMessage(
            ValueError, "generateArbitraryPatch: only rotation around z-axis",
        ):
            arrRot = self.arrRot
            arrRot[1, 0] = 1
            generateArbitraryPatch(
                self.arrPos, arrRot, self.arrSize, self.arrFreq1,
            )

    def test_value(self):

        EImport = fromArraydefData(self.matlabEADFPath)
        ECreate = generateArbitraryPatch(
            self.arrPos[:, :2],
            self.arrRot[:, :2],
            self.arrSize[:, :2],
            self.arrFreq1,
            addCrossPolPort=True,
        )
        arrAzi = np.linspace(-np.pi, np.pi, 21, endpoint=True)
        arrCoEle = np.linspace(0, np.pi, 11, endpoint=True)
        grdAzi, grdCoEle = toGrid(arrAzi, arrCoEle)
        # calculate pattern at given angles
        patternImport = EImport.pattern(grdAzi, grdCoEle)
        patternCreate = ECreate.pattern(grdAzi, grdCoEle)

        # only check one port. must be the first one. this one was not
        # rotated.
        self.assertTrue(
            np.allclose(patternImport[..., 0], patternCreate[..., 0])
        )


if __name__ == "__main__":
    unittest.main()
