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
from unittest.mock import patch
import os
import numpy as np

from eadf.arrays import *
from eadf.auxiliary import *
from eadf.backend import *
from eadf.importers import *

from . import TestCase


class TestFromAngleListData(TestCase):
    def setUp(self):
        self.array = generateUCA(13, 1.2, np.array([2]), np.linspace(4, 4, 1))
        self.arrCoEleS, self.arrAziS = toGrid(*sampleAngles(10, 20))
        self.arrCoEleI, self.arrAziI = toGrid(*sampleAngles(30, 60))

        self.dataS = self.array.pattern(self.arrCoEleS, self.arrAziS)

    def test_inputSizeFail1(self):
        with self.assertRaisesWithMessage(
            ValueError,
            (
                "fromAngleListData: Input arrays"
                + " of sizes %d ele, %d azi, %d values dont match"
            )
            % (
                self.arrCoEleS.shape[0],
                self.arrAziS[:-1].shape[0],
                self.dataS.shape[0],
            ),
        ):
            fromAngleListData(
                self.arrCoEleS,
                self.arrAziS[:-1],
                self.dataS,
                self.array.arrFreq,
                self.array.arrPos,
                30,
                60,
            )

    def test_inputSizeFail2(self):
        with self.assertRaisesWithMessage(
            ValueError,
            (
                "fromAngleListData: Input arrays"
                + " of sizes %d ele, %d azi, %d values dont match"
            )
            % (
                self.arrCoEleS[:-1].shape[0],
                self.arrAziS[:-1].shape[0],
                self.dataS.shape[0],
            ),
        ):
            fromAngleListData(
                self.arrCoEleS[:-1],
                self.arrAziS[:-1],
                self.dataS,
                self.array.arrFreq,
                self.array.arrPos,
                30,
                60,
            )

    def test_inputSizeFail3(self):
        with self.assertRaisesWithMessage(
            ValueError,
            (
                "fromAngleListData:"
                + "Number of positions %d does not match provided data %d"
            )
            % (self.array.arrPos.shape[1] - 1, self.dataS.shape[3]),
        ):
            fromAngleListData(
                self.arrCoEleS,
                self.arrAziS,
                self.dataS,
                self.array.arrFreq,
                self.array.arrPos[:, :-1],
                30,
                60,
            )

    def test_inputSizeFail4(self):
        with self.assertRaisesWithMessage(
            ValueError,
            (
                "fromAngleListData:"
                + "Number of freqs %d does not match provided data %d"
            )
            % (self.array.arrFreq.shape[0] - 1, self.dataS.shape[1]),
        ):
            fromAngleListData(
                self.arrCoEleS,
                self.arrAziS,
                self.dataS,
                self.array.arrFreq[:-1],
                self.array.arrPos,
                30,
                60,
            )

    def test_inputSizeFail5(self):
        with self.assertRaisesWithMessage(
            ValueError, "fromAngleListData: numAzi must be larger than 0."
        ):
            fromAngleListData(
                self.arrCoEleS,
                self.arrAziS,
                self.dataS,
                self.array.arrFreq,
                self.array.arrPos,
                30,
                -60,
            )

    def test_inputSizeFail6(self):
        with self.assertRaisesWithMessage(
            ValueError, "fromAngleListData: numCoEle must be larger than 0."
        ):
            fromAngleListData(
                self.arrCoEleS,
                self.arrAziS,
                self.dataS,
                self.array.arrFreq,
                self.array.arrPos,
                -30,
                60,
            )

    def test_hfss_import(self):
        with self.assertRaisesWithMessage(
            ValueError, "fromAngleListData: numCoEle must be larger than 0."
        ):
            fromAngleListData(
                self.arrCoEleS,
                self.arrAziS,
                self.dataS,
                self.array.arrFreq,
                self.array.arrPos,
                -30,
                60,
            )


class TestMatlabImporters(TestCase):
    def setUp(self):
        # path to the right Matlab EADF
        dirname = os.path.dirname(__file__)
        self.correctPath = os.path.join(
            dirname, "data/NarrowBandPatchArray.mat"
        )
        self.wrongPath = os.path.join(dirname, "data/wrongPath.mat")
        self.wrongFile = os.path.join(dirname, "data/DummyMatfile.mat")

    def test_inputPathFail(self):
        with self.assertRaisesWithMessage(
            IOError,
            "fromArraydefData: file %s does not exist." % (self.wrongPath,),
        ):
            fromArraydefData(self.wrongPath)

    def test_keyError(self):
        with self.assertRaisesWithMessage(
            KeyError, "fromArraydefData: key 'arraydef' does not exist."
        ):
            fromArraydefData(self.wrongFile)

    def test_value(self):
        EImport = fromArraydefData(self.correctPath)
        ECreate = generateArbitraryPatch(
            np.stack((np.array([0, 0, 0]), np.array([0, 0, 35e-3])), axis=0).T,
            np.zeros((3, 2)),
            np.repeat(np.array([[30e-3, 30e-3, 1e-3]]).T, 2, axis=1),
            np.array([5e9]),
            addCrossPolPort=True,
        )
        arrCoEle, arrAzi = EImport.arrCoEle, EImport.arrAzi
        grdCoEle, grdAzi = toGrid(arrCoEle, arrAzi)
        # calculate pattern at given angles
        patternImport = EImport.pattern(grdCoEle, grdAzi)
        patternCreate = ECreate.pattern(grdCoEle, grdAzi)

        # only check one port. must be the first one. this one was not
        # rotated.
        self.assertTrue(
            np.allclose(patternImport[..., 0], patternCreate[..., 0])
        )


if __name__ == "__main__":
    unittest.main()
