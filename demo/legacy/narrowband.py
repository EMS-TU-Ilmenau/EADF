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
Wideband Array Handling
-----------------------

This demo first off shows, how to use pickling in production and second
how the interpolation in excitation frequency works.

The pickling is used here, since the preprocessing for the interpolation
can take some time.

>>> cd demo
>>> python narrowband.py


This then should yield something like the following.

.. figure:: _static/demo_narrowband.png

  Output for some exemplatory measurement data.
"""

import eadf
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# handle these with care!
numAzi = 180
numEle = 90
numFreq = 3


def loadData(path: str) -> tuple:
    r"""
    This function serves as a simple importer to get the data from disk
    into the format that is used by the EADF. Here you are on your own,
    since your data might be rearranged differently.

    The tuple that is returned should contain the beam pattern,
    angular values and frequency values.
    """
    data = scipy.io.loadmat(path)

    # this is just stuff brought to you by matlab
    arrCoEle = (
        xp.radians(data["pattern"][0][0][0][0][0][1].flatten()) + xp.pi / 2
    )
    arrAzi = xp.radians(data["pattern"][0][0][0][0][1][1].flatten())
    arrFreq = data["pattern"][0][0][0][0][4][1]
    arrPattern = data["pattern"][0][0][1]

    # bring the data in the EADF data format
    arrPattern = xp.swapaxes(arrPattern, 2, 4)

    return (arrPattern, arrCoEle, arrAzi, arrFreq)


if __name__ == "__main__":
    # load the data and some meta data
    arrPattern, arrCoEle, arrAzi, arrFreq = loadData("pattern_IZTR5509.mat")

    array = eadf.EADF(
        arrPattern[:, :, :, :, :2],
        arrCoEle,
        arrAzi[:],
        arrFreq[:],
        xp.random.randn(3, 2),
        keepNarrowBand=True,
    )

    # generate interpolation parameter
    pltAzi, pltCoEle = eadf.auxiliary.sampleAngles(
        numEle, numAzi, lstEndPoints=[0, 1]
    )

    # generate a 3D grid of angles and frequencies
    grdAzi, grdCoEle = eadf.auxiliary.toGrid(pltAzi, pltCoEle)

    a1 = array.pattern(grdCoEle, grdAzi, arrFreq[0]).reshape(
        (numEle, numAzi, 2, 2)
    )
    a2 = array.pattern(grdCoEle, grdAzi, arrFreq[15]).reshape(
        (numEle, numAzi, 2, 2)
    )
    a3 = array.pattern(grdCoEle, grdAzi, arrFreq[-1]).reshape(
        (numEle, numAzi, 2, 2)
    )

    a1 = eadf.asNumpy(a1)

    plt.subplot(321)
    plt.imshow(xp.abs(a1[:, :, 1, 0]))
    plt.colorbar()
    plt.subplot(322)
    plt.imshow(xp.abs(arrPattern[:, :, 0, 1, 0]))
    plt.colorbar()
    plt.subplot(323)
    plt.imshow(xp.abs(a2[:, :, 1, 0]))
    plt.colorbar()
    plt.subplot(324)
    plt.imshow(xp.abs(arrPattern[:, :, 15, 1, 0]))
    plt.colorbar()
    plt.subplot(325)
    plt.imshow(xp.abs(a3[:, :, 1, 0]))
    plt.colorbar()
    plt.subplot(326)
    plt.imshow(xp.abs(arrPattern[:, :, -1, 1, 0]))
    plt.colorbar()
    plt.show()
