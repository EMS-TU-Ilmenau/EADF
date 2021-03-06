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
Operating Frequencies
---------------------

This demo shows how to work with antenna arrays that contain frequency
dependent measurement data.

Terminology of the Package
^^^^^^^^^^^^^^^^^^^^^^^^^^

 - *measurement/input data*: this data contains samples in CoElevation x
   Azimuth x Frequency x Polarization x Port. Here one hopes that these
   samples describe the behaviour of the antenna well enough to be considered
   accurate. This is usually the data we will feed into the constructor of the
   EADF object, along with some metadata.
 - *narrowband*: this refers to the case, where we assume that the samples in
   frequency are not so widely spaced such that the behaviour of the antenna
   changes significantly over the specified frequencies
 - *wideband*: this case is reflected in the fact that the frequency bins
   cover such a wide range that the antenna shows very different behaviour for
   different frequencies.
 - *stationary subband*: this defines a region in frequency over which we can
   describe the antenna response reasonably well with one single set of CoEle
   x Azi x Pol x Port data values (not necessarily measurement data, it can be
   processed or calculated).
 - *operating frequencies*: if used in an estimator, one usually has the
   scenario, where one or several antennas were deployed to take measurements
   as discrete frequencies occupying a certain band, thus using a certain
   bandwidth. These frequencies are the so called operating frequencies.
   Generally, these are *independent* of the frequencies (still they have to
   be resided between the lowest and highest frequency present in the
   measurement data) used in the measurement data that describe the antenna as
   a system. For the estimator one needs the antenna response at the operating
   frequencies.
 - *stationary operating subbands*: during parameter estimation from
   measurement data speed and efficiency are critical. If the operating
   frequencies are in a stationary subband, we group these to a so called
   stationary operating subband. This saves precious computation time, since
   several frequencies can be treated with a single beampattern calculation.

The Different "Types" of Frequencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is now clear that we have *two* sets of frequencies. Those present in the
measurement data and those needed during estimation, moreover the whole
bandwith of the data is divided up in stationary subbands. The job of the
package now is to use the measurement data in order to provide the division
into stationary subbands (automatically or manually) and to provide the angle
dependent antenna response at the operating frequencies while exploiting the
previously derived stationarity along certain subbands. By default the EADF
object assumes *no* stationary operating subbands and the operating
frequencies are the frequencies provided in the measurement data. If the user
wants different operating frequencies, the respective beampatterns are
generated by means of spline interpolation along this measurement dimension.

This Demo
^^^^^^^^^

This script first generates two EADF objects that represent the same antenna,
but with a different sampling resolution in frequency. The antenna with a
lower resolution is used to interpolate the missing values and is compared
to the antenna sampled at higher resolution. Finally, we also define
an array which shows how to make use of the stationary subbands.

>>> python operfreq.py

.. figure:: _static/demo_operfreq.png

  Output of this script.
"""

if __name__ == "__main__":
    import eadf
    from eadf.backend import xp, asNumpy
    import matplotlib.pyplot as plt

    numFreq = 33
    indCoEle = 44
    indAzi = 0
    f0 = 120 * 1e4
    f1 = 120 * 1e8
    numSize = 0.1

    # coarsely sampeled array
    # with only a few frequency bins
    arrayLowRes = eadf.generateURA(
        1,
        1,
        0.5,
        0.75,
        numSize * xp.ones((3, 1)),
        xp.linspace(f0, f1, (numFreq + 1) // 4),
    )

    # densely sampled array
    # with "all" frequency bins
    # this will be used as a comparison to arrayLowRes
    arrayHighRes = eadf.generateURA(
        1,
        1,
        0.5,
        0.75,
        numSize * xp.ones((3, 1)),
        xp.linspace(f0, f1, numFreq),
    )

    # subband array
    arraySubBand = eadf.generateURA(
        1,
        1,
        0.5,
        0.75,
        numSize * xp.ones((3, 1)),
        xp.linspace(f0, f1, numFreq),
    )

    # average over pairs of 5 to get some nice fourier data for each
    # of the bands
    arrSubBandData = xp.empty_like(arrayHighRes.arrFourierData[:, :, ::5])
    for ii in range(arrSubBandData.shape[2]):
        arrSubBandData[:, :, ii] = xp.mean(
            arrayHighRes.arrFourierData[:, :, ii * 5 : (ii + 1) * 5 + 2],
            axis=2,
        )

    # here the subbands are defined where roughly 5 frequencies are
    # collected in one subband.
    arraySubBand.defineSubBands(
        xp.linspace(f0, f1, arrayHighRes.arrFourierData[:, :, ::5].shape[2]),
        arrSubBandData,
    )

    # now introduce a more densely sampled frequency grid for the array where we
    # only have coarsely sampled data available
    # this triggers spline interpolation along frequencies to guess
    # the pattern data for the newly set operating frequencies
    arrayLowRes.operatingFrequencies = xp.linspace(f0, f1, numFreq)

    # sample the low resolution array at the newly defined operating
    # frequencies
    patternLowRes = asNumpy(
        arrayLowRes.pattern(
            xp.array([arrayLowRes.arrCoEle[indCoEle]]),
            xp.array([arrayLowRes.arrAzi[indAzi]]),
        )
    )

    # sample the subband array
    patternSubBand = asNumpy(
        arraySubBand.pattern(
            xp.array([arrayLowRes.arrCoEle[indCoEle]]),
            xp.array([arrayLowRes.arrAzi[indAzi]]),
        )
    )

    plt.subplot(211)
    plt.title("real")

    for sb in arraySubBand.subBandIntervals:
        plt.axvline(sb, color="grey")

    plt.scatter(
        asNumpy(arrayLowRes.arrFreq),
        asNumpy(xp.real(arrayLowRes.arrRawData[indCoEle, indAzi, :, 0, 0])),
        marker="o",
        color="orange",
        s=150,
        label="used samples",
    )
    plt.scatter(
        asNumpy(arrayHighRes.arrFreq),
        asNumpy(xp.real(arrayHighRes.arrRawData[indCoEle, indAzi, :, 0, 0])),
        marker="*",
        color="red",
        zorder=10,
        label="true values",
    )
    plt.plot(
        asNumpy(arrayLowRes._operatingFrequencies),
        asNumpy(xp.real(patternLowRes[0, :, 0, 0])),
        marker="o",
        color="blue",
        label="inferred values",
    )
    plt.plot(
        asNumpy(arraySubBand._operatingFrequencies),
        asNumpy(xp.real(patternSubBand[0, :, 0, 0])),
        color="green",
        marker="x",
        label="subband values",
    )
    plt.legend()
    plt.subplot(212)
    plt.title("imag")

    for sb in arraySubBand.subBandIntervals:
        plt.axvline(sb, color="grey")

    plt.scatter(
        asNumpy(arrayLowRes.arrFreq),
        asNumpy(xp.imag(arrayLowRes.arrRawData[indCoEle, indAzi, :, 0, 0])),
        marker="o",
        color="orange",
        s=150,
    )
    plt.scatter(
        asNumpy(arrayHighRes.arrFreq),
        asNumpy(xp.imag(arrayHighRes.arrRawData[indCoEle, indAzi, :, 0, 0])),
        marker="*",
        color="red",
        zorder=10,
    )
    plt.plot(
        asNumpy(arrayLowRes._operatingFrequencies),
        asNumpy(xp.imag(patternLowRes[0, :, 0, 0])),
        marker="o",
        color="blue",
    )
    plt.plot(
        asNumpy(arraySubBand._operatingFrequencies),
        asNumpy(xp.imag(patternSubBand[0, :, 0, 0])),
        color="green",
        marker="x",
        label="subband values",
    )
    plt.show()
