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
Parse arraydef from .mat-file
-----------------------------

This demo parses an already existing EADF from a .mat-file

>>> cd demo
>>> python parseArraydef.py

This then should yield something like the following.

.. figure:: _static/demo_parseArraydef.png

  Output for some exemplatory measurement data.
"""


import eadf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    E = eadf.importers.fromArraydefData("TUI_SPUCA2x8_I_20MHz.mat")

    arrAzi = E.arrAzi / xp.pi * 180
    arrCoEle = E.arrCoEle / xp.pi * 180
    arrEle = 90 - arrCoEle

    # plot results
    ax1 = plt.subplot(131)
    plt.imshow(xp.abs(E.arrRawData[:, :, 0, 0, 0]))
    ax1.set_title("Periodified BP")
    ax2 = plt.subplot(132)
    plt.imshow(xp.fft.fftshift(xp.abs(E.arrRawFourierData[:, :, 0, 0, 0])))
    ax2.set_title("Fourier coefficients")
    ax3 = plt.subplot(133)
    plt.imshow(
        xp.abs(E.arrRawData[: arrCoEle.shape[0], :, 0, 0, 0]),
        extent=[arrAzi[0], arrAzi[-1], arrEle[0], arrEle[-1]],
    )
    ax3.set_title("Beampattern")
    plt.show()
