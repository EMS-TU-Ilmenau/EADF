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
Array Generation
----------------

This demo shows the generation of (dual-polarimetric) uniform linear array
based on either dipol or patch elements with the delivered array functions of
this package.

>>> cd demo
>>> python arrayGeneration.py

This then should yield two plots with beampattern of a patch and a dipole array.

.. figure:: _static/demo_inputviz.png

  Output for some exemplatory measurement data.
"""


import scipy.constants
import matplotlib.pyplot as plt
import numpy as np
import eadf


def generateSPUCA(numFreq) -> eadf.EADF:
    r"""
    This function generates a stacked polarimetric uniform circular (dipole)
    array
    (SPUCA) with 2 rings and 8 dual-polarimetric elements per ring for the
    given frequency. The array has 32 elements in total. Each dipole element
    will have a length of lambda/4, the radius will be lambda/2 as well as the
    distance between the rings.

    The EADF of this array is returned
    """

    wavelength = scipy.constants.c / numFreq
    dipoleLength = xp.array([wavelength / 4])
    arrFreq = xp.array([numFreq])

    return eadf.arrays.generateStackedUCA(
        8,
        2,
        wavelength / 2,
        wavelength / 2,
        dipoleLength,
        arrFreq,
        addCrossPolPort=True,
    )


def generateSPUCPA(numFreq) -> eadf.EADF:
    r"""
    This function generates a stacked polarimetric uniform circular patch array
    (SPUCPA) with 2 rings and 8 dual-polarimetric elements per ring for the
    given frequency. The array has 32 elements in total. Each patch element will
    have a size of (lambda/4 x lambda/4 x 1mm), the radius will be lambda/2 as
    well as the distance between the rings.

    The EADF of this array is returned
    """

    wavelength = scipy.constants.c / numFreq
    patchSize = xp.array([[wavelength / 4, wavelength / 4, 1e-3]]).T
    arrFreq = xp.array([numFreq])

    return eadf.arrays.generateStackedUCA(
        8,
        2,
        wavelength / 2,
        wavelength / 2,
        patchSize,
        arrFreq,
        addCrossPolPort=True,
    )


if __name__ == "__main__":
    f = 2.4e9
    # generate EADFs of patch and dipole arrays, respectively
    EDipole = generateSPUCA(f)
    EPatch = generateSPUCPA(f)

    arrAzi = xp.linspace(-xp.pi, xp.pi, 61, endpoint=True)
    arrCoEle = xp.linspace(0, xp.pi, 31, endpoint=True)

    gridAzi, gridCoEle = eadf.auxiliary.toGrid(arrAzi, arrCoEle)

    patternDipole = EDipole.pattern(gridCoEle, gridAzi, f)
    patternPatch = EPatch.pattern(gridCoEle, gridAzi, f)

    # some fancy plots of the different beampatterns
    # plot of dipole array pattern
    fig = plt.figure()
    fig.suptitle("Dipole Array", fontsize=16)
    ax1 = plt.subplot(221)
    plt.pcolor(
        arrAzi / xp.pi * 180,
        arrCoEle / xp.pi * 180,
        xp.reshape(
            xp.abs(patternDipole[:, 0, 0]),
            (arrCoEle.shape[0], arrAzi.shape[0]),
        ),
    )
    plt.gca().invert_yaxis()
    plt.colorbar()
    ax1.set_title("Element 1 (front, Pol=H) EPhi")
    ax1.set_xlabel("Azimuth [deg]")
    ax1.set_ylabel("Co-Ele[deg]")

    ax1 = plt.subplot(222)
    plt.pcolor(
        arrAzi / xp.pi * 180,
        arrCoEle / xp.pi * 180,
        xp.reshape(
            xp.abs(patternDipole[:, 0, 3]),
            (arrCoEle.shape[0], arrAzi.shape[0]),
        ),
    )
    plt.gca().invert_yaxis()
    plt.colorbar()
    ax1.set_title("Element 4 (side, Pol=H) EPhi")
    ax1.set_xlabel("Azimuth [deg]")
    ax1.set_ylabel("Co-Ele[deg]")

    ax1 = plt.subplot(223)
    plt.pcolor(
        arrAzi / xp.pi * 180,
        arrCoEle / xp.pi * 180,
        xp.reshape(
            xp.abs(patternDipole[:, 1, 16]),
            (arrCoEle.shape[0], arrAzi.shape[0]),
        ),
    )
    plt.gca().invert_yaxis()
    plt.colorbar()
    ax1.set_title("Element 17 (front, Pol=V) ETheta")
    ax1.set_xlabel("Azimuth [deg]")
    ax1.set_ylabel("Co-Ele[deg]")

    ax1 = plt.subplot(224)
    plt.pcolor(
        arrAzi / xp.pi * 180,
        arrCoEle / xp.pi * 180,
        xp.reshape(
            xp.abs(patternDipole[:, 1, 19]),
            (arrCoEle.shape[0], arrAzi.shape[0]),
        ),
    )
    plt.gca().invert_yaxis()
    plt.colorbar()
    ax1.set_title("Element 20 (side, Pol=V) ETheta")
    ax1.set_xlabel("Azimuth [deg]")
    ax1.set_ylabel("Co-Ele[deg]")

    #############################
    # plot of patch array pattern
    #############################
    fig = plt.figure()
    fig.suptitle("Patch Array", fontsize=16)
    ax1 = plt.subplot(221)
    plt.pcolor(
        arrAzi / xp.pi * 180,
        arrCoEle / xp.pi * 180,
        xp.reshape(
            xp.abs(patternPatch[:, 0, 0]),
            (arrCoEle.shape[0], arrAzi.shape[0]),
        ),
    )
    plt.gca().invert_yaxis()
    plt.colorbar()
    ax1.set_title("Element 1 (front, Pol=H) EPhi")
    ax1.set_xlabel("Azimuth [deg]")
    ax1.set_ylabel("Co-Ele[deg]")

    ax1 = plt.subplot(222)
    plt.pcolor(
        arrAzi / xp.pi * 180,
        arrCoEle / xp.pi * 180,
        xp.reshape(
            xp.abs(patternPatch[:, 0, 3]),
            (arrCoEle.shape[0], arrAzi.shape[0]),
        ),
    )
    plt.gca().invert_yaxis()
    plt.colorbar()
    ax1.set_title("Element 4 (side, Pol=H) EPhi")
    ax1.set_xlabel("Azimuth [deg]")
    ax1.set_ylabel("Co-Ele[deg]")

    ax1 = plt.subplot(223)
    plt.pcolor(
        arrAzi / xp.pi * 180,
        arrCoEle / xp.pi * 180,
        xp.reshape(
            xp.abs(patternPatch[:, 1, 16]),
            (arrCoEle.shape[0], arrAzi.shape[0]),
        ),
    )
    plt.gca().invert_yaxis()
    plt.colorbar()
    ax1.set_title("Element 17 (front, Pol=V) ETheta")
    ax1.set_xlabel("Azimuth [deg]")
    ax1.set_ylabel("Co-Ele[deg]")

    ax1 = plt.subplot(224)
    plt.pcolor(
        arrAzi / xp.pi * 180,
        arrCoEle / xp.pi * 180,
        xp.reshape(
            xp.abs(patternPatch[:, 1, 19]),
            (arrCoEle.shape[0], arrAzi.shape[0]),
        ),
    )
    plt.gca().invert_yaxis()
    plt.colorbar()
    ax1.set_title("Element 20 (side, Pol=V) ETheta")
    ax1.set_xlabel("Azimuth [deg]")
    ax1.set_ylabel("Co-Ele[deg]")

    plt.show()
