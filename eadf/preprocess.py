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
Preprocessing Methods
---------------------

This module hosts several preprocessing methods that can be used during
and before construction of an EADF object.
"""


from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np

__all__ = [
    "sampledToFourier",
    "setCompressionFactor",
    "splineInterpolateFrequency",
    "symmetrizeData",
    "regularSamplingToGrid",
]


def sampledToFourier(arrData: np.ndarray) -> tuple:
    """Transform the regularly sampled data in frequency domain

    Here we assume that the data is already flipped along co-elevation,
    rotated along azimuth as described in the EADF paper and in the wideband
    case it is also periodified in excitation frequency direction such that we
    can just calculate the respective 2D/3D FFT from this along the first two
    /three axes.

    Parameters
    ----------
    data : np.ndarray
        Raw sampled and periodified data in the form
        2 * co-ele x azi x freq x pol x elem

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        Fourier Transform and the respective sample frequencies

    """
    axes = [0, 1]

    freqs = tuple(
        (arrData.shape[ii] * np.fft.fftfreq(arrData.shape[ii])) for ii in axes
    )

    scaling = 1.0
    for aa in axes:
        scaling *= arrData.shape[aa]

    # the frequencies are generated according to (5) in the EADF paper
    res = np.fft.fft2(arrData, axes=axes) / scaling

    return (res, *freqs)


def setCompressionFactor(
    arrFourierData: np.ndarray,
    numCoEleInit: int,
    numAziInit: int,
    numValue: float,
) -> tuple:
    """Calculate Subselection-Indices

    This method takes the supplied compression factor, which is
    not with respect to the number of Fourier coefficients to use
    but rather the amount of energy still present in them. this is
    achieved by analysing the spatial spectrum of the whole array in the
    following way:

      1. Flip the spectrum all into one quadrant
      2. normalize it with respect to the complete energy
      3. find all combinations of subsizes in azimuth and elevation
         such that the energy is lower than numValue
      4. find the pair of elevation and azimuth index such that it
         minimizes the execution time during sampling

    Parameters
    ----------
    arrFourierData : np.ndarray
        the description of the array in frequency domain.
    numCoEleInit : int
        number of coelevation samples
    numAziInit : int
        number of azimuth samples
    numValue : float
        Compression Factor we would like to have

    Returns
    -------
    tuple
        compression factor and subselection indices.

    """

    # calculate the energy of the whole array
    # first get the norm of each spectrum, then sum along
    # pol, freq and elem
    numTotalEnergy = np.linalg.norm(arrFourierData)

    # find the middle indices in the first two components
    middleCoEle = int((numCoEleInit + (numCoEleInit % 2)) / 2)
    middleAzi = int((numAziInit + (numAziInit % 2)) / 2)

    middlePadCoEle = int(numCoEleInit % 2)
    middlePadAzi = int(numAziInit % 2)

    tplMiddle = (middleCoEle, middleAzi)
    tplInit = (numCoEleInit, numAziInit)

    # first we sum along polarisation and the array elements
    arrFoldedData = np.sum(np.abs(arrFourierData ** 2), axis=(2, 3, 4))

    # then we flip and add everything for each of the 2 dimensions
    arrFoldedData = arrFoldedData[:middleCoEle] + np.pad(
        arrFoldedData[middleCoEle:][::-1],
        ((0, middlePadCoEle), (0, 0)),
        mode="constant",
    )
    arrFoldedData = arrFoldedData[:, :middleAzi] + np.pad(
        arrFoldedData[:, middleAzi:][:, ::-1],
        ((0, 0), (0, middlePadAzi)),
        mode="constant",
    )

    # get the partial norms for all the combinations
    arrCumulative = (
        np.sqrt(np.cumsum(np.cumsum(arrFoldedData, axis=0), axis=1))
        / numTotalEnergy
    )

    # find all subarrays such that they have less energy than required
    # the others will be the ones that we might select from
    arrInFeasibleSizes = arrCumulative < numValue - 1e-14

    # cost function is just the outer product of the three indexes, since
    # we have to essentially do linear time operations when
    # sampling
    arrCost = np.einsum(
        "i,j->ij",
        np.linspace(1, 2 * middleCoEle, middleCoEle),
        np.linspace(1, 2 * middleAzi, middleAzi),
    )

    # all infeasible indices will be set to infinity, such that
    # they are not considered when finding the optimal cost indices
    arrCost[arrInFeasibleSizes] = np.inf

    # find the minimum in the 2D array of cost values
    minCostInd = np.unravel_index(np.argmin(arrCost, axis=None), arrCost.shape)

    # find the actual compression factor
    compressionFactor = arrCumulative[minCostInd]

    # now get the actual subselection arrays as booleans
    # here we have to apply the inverse fftshift
    # we have the +1, since minCostInd contains indices starting at
    # 0, but the linspace thingy is done from "1"
    arrIndCompress = (
        np.fft.ifftshift(
            np.abs(np.linspace(-tplMiddle[ii], +tplMiddle[ii], tplInit[ii]))
            <= mci + 1
        )
        for ii, mci in enumerate(minCostInd)
    )

    res = tuple([*arrIndCompress])

    # return the subselection indices we determined
    return res


def splineInterpolateFrequency(
    arrFreq: np.ndarray, arrData: np.ndarray
) -> np.ndarray:
    """Spline Interpolation of Pattern Data in Frequency

    Splines!

    Parameters
    ----------
    arrData : np.ndarray
        data to pad
    Returns
    -------
    np.ndarray
        of interpolation splines
    """

    # create a numpy array of splines
    arrSplines = np.empty(
        (*arrData.shape[:2], 2, *arrData.shape[3:]), dtype=object
    )

    for ii0 in range(arrData.shape[0]):
        for ii1 in range(arrData.shape[1]):
            for ii3 in range(arrData.shape[3]):
                for ii4 in range(arrData.shape[4]):
                    # extract the data
                    y = arrData[ii0, ii1, :, ii3, ii4]

                    # real part spline
                    arrSplines[
                        ii0, ii1, 0, ii3, ii4
                    ] = InterpolatedUnivariateSpline(arrFreq, np.real(y), k=5)

                    # imaginary part spline
                    arrSplines[
                        ii0, ii1, 1, ii3, ii4
                    ] = InterpolatedUnivariateSpline(arrFreq, np.imag(y), k=5)

    return arrSplines


def symmetrizeData(arrA: np.ndarray) -> np.ndarray:
    """Generate a symmetrized version of a regularly sampled array data

    This function assumes that we are given the beam pattern sampled in
    co-elevation and azimuth on a regular grid, as well as for at most 2
    polarizations and all the same wave-frequency bins. Then this function
    applies (2) in the original EADF paper. So the resulting array has
    the same dimensions but 2*n-1 the size in co-elevation direction, if
    n was the original co-elevation size.

    Parameters
    ----------
    arrA : np.ndarray
        Input data (co-elevation x azimuth x pol x freq x elem).

    Returns
    -------
    np.ndarray
        Output data (2*co-elevation - 2 x azimuth x pol x freq x elem).

    """
    if len(arrA.shape) != 5:
        raise ValueError(
            "symmetrizeData: got %d dimensions instead of 5"
            % (len(arrA.shape))
        )

    # allocate memory
    arrRes = np.tile(arrA, (2, 1, 1, 1, 1))[:-2]

    # Equation (2) in EADF Paper by Landmann and DGO
    # or more correctly the equations (3.13) - (3.17) in the
    # dissertation of Landmann
    arrRes[arrA.shape[0] :] = -np.roll(
        np.flip(arrA[1:-1], axis=0),
        shift=int((arrA.shape[1] - (arrA.shape[1] % 2)) / 2),
        axis=1,
    )

    # azi in [0, 2pi] -> [-pi, +pi]
    # return arrRes
    return np.fft.fftshift(arrRes, axes=(0,))


def regularSamplingToGrid(
    arrA: np.ndarray, numCoEle: int, numAzi: int
) -> np.ndarray:
    """Reshape an array sampled on a 2D grid to actual 2D data

    Parameters
    ----------
    arrA : np.ndarray
        Input data `arrA` (2D angle x pol x freq x elem).
    numCoEle : int
        Number of samples in co-elevation direction.
    numAzi : int
        Number of samples in azimuth direction.

    Returns
    -------
    np.ndarray
        Output data (co-elevation x azimuth x freq x pol x elem).

    """
    if arrA.shape[0] != (numAzi * numCoEle):
        raise ValueError(
            (
                "regularSamplingToGrid:"
                + "numCoEle %d, numAzi %d and arrA.shape[0] %d dont match"
            )
            % (numAzi, numCoEle, arrA.shape[0])
        )
    if len(arrA.shape) != 4:
        raise ValueError(
            (
                "regularSamplingToGrid:"
                + "Input arrA has %d dimensions instead of 4"
            )
            % (len(arrA.shape))
        )

    return arrA.reshape((numCoEle, numAzi, *arrA.shape[1:]))
