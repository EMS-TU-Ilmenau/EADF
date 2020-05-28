# Copyright 2019 EMS Group TU Ilmenau
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
Importing Wideband Data
-----------------------

Demo to import wideband pattern data stored as .mat file to produce an EADF
object

>>> cd demo
>>> python patternWideBandImport.py
"""

import eadf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # name of pattern .mat file
    fileNamePattern = "pattern_IZTR5509.mat"
    # select first antenna port - index 0
    arrPorInd = xp.array([0])
    # select vertical polarization
    arrPol = xp.array(["v"])
    # select frequency points
    arrFreq = xp.array([4.0e8, 5.0e8])

    kwargs = {
        "arrPorIndUser": arrPorInd,
        "arrPolUser": arrPol,
        "arrFreqUser": arrFreq,
    }

    # obtain measured pattern for chosen paramters and corresponding
    # EADF object
    EADF = eadf.importers.fromWidebandAngData(fileNamePattern, **kwargs)

    # set lowMemory flag
    EADF.lowMemory = True

    numCoEle = 30
    numAzi = 60
    numPol = arrPol.size
    numPor = arrPorInd.size
    numFreq = arrFreq.size

    pltCoEle, pltAzi = eadf.auxiliary.sampleAngles(
        numCoEle, numAzi, lstEndPoints=[0, 1]
    )
    pltFreq = xp.linspace(arrFreq[0], arrFreq[-1], numFreq, endpoint=True)

    # generate a 3D grid of angles and frequencies
    grdCoEle, grdAzi, grdFreq = eadf.auxiliary.toGrid(
        pltCoEle, pltAzi, pltFreq
    )

    # pattern obtained using EADF
    patternEADF = EADF.pattern(grdCoEle, grdAzi, grdFreq).reshape(
        (numCoEle, numAzi, numFreq, numPol, numPor), order="F"
    )

    f = plt.figure()

    arrCoEle = xp.linspace(
        0, 180, int(EADF.arrRawData.shape[0] / 2), endpoint=False
    )
    arrAzi = xp.linspace(
        -180, 180, xp.ma.size(EADF.arrRawData, 1), endpoint=False
    )

    ax1 = plt.subplot(211)
    plt.pcolor(
        arrAzi,
        arrCoEle,
        xp.abs(
            EADF.arrRawData[: int(EADF.arrRawData.shape[0] / 2), :, 0, 0, 0]
        ),
    )
    plt.gca().invert_yaxis()
    plt.colorbar()
    ax1.set_title("Measured Beampattern")
    ax1.set_xlabel("Azimuth [deg.]")
    ax1.set_ylabel("Co-elevation [deg.]")

    arrAzi = xp.linspace(-180, 180, numAzi, endpoint=False)
    arrCoEle = xp.linspace(0, 180, numCoEle, endpoint=False)

    ax1 = plt.subplot(212)
    plt.pcolor(arrAzi, arrCoEle, xp.abs(patternEADF[:, :, 0, 0, 0]))
    plt.gca().invert_yaxis()
    plt.colorbar()
    ax1.set_title("Interpolated Beampattern ")
    ax1.set_xlabel("Azimuth [deg.]")
    ax1.set_ylabel("Co-elevation [deg.]")

    f.tight_layout(pad=1.0)
    plt.show()
