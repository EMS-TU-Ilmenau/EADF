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

from scipy.special import sph_harm
import numpy as np

__all__ = ["interpolateDataSphere"]


def interpolateDataSphere(
    arrCoEleSample: np.ndarray,
    arrAziSample: np.ndarray,
    arrValues: np.ndarray,
    arrCoEleInter: np.ndarray,
    arrAziInter: np.ndarray,
    **options
) -> np.ndarray:
    """Interpolate Data located on a Sphere

    This method can be used for interpolating a function of the form
    f : S^2 -> C which is sampled on N arbitrary positions on the sphere.
    The input data is assumed to be in the format N x M1 x ... and the
    interpolation is broadcasted along M1 x ...
    The interpolation is always done using least squares, so for noisy data
    or overdetermined data with respect to the basis you should not encounter
    any problems.

    *Methods*
     - *SH* (Spherical Harmonics), see dissertation delGaldo,
       For these you have to supply *numN*
       as a kwarg, which determines the order of the SH basis. The number
       of total basis functions is then calculated via
       *numN x (numN + 1) + 1*. default=6

    Parameters
    ----------
    arrCoEleSample : np.ndarray
        Sampled Co-Elevation positions in radians
    arrAziSample : np.ndarray
        Sampled Azimuth positions in radians
    arrValues : np.ndarray
        Sampled values
    arrCoEleInter : np.ndarray
        CoElevation positions we want the function to be evaluated in radians
    arrAziInter : np.ndarray
        Azimuth Positions we want the function to be evaluated in radians
    method : type, optional, default='SH'
        'SH'(default) for spherical harmonics
    **options : type
        Depends on method, see above

    Returns
    -------
    np.ndarray
        Description of returned object.

    """

    method = options.get("method", "SH")

    if (
        (arrAziSample.shape[0] != arrCoEleSample.shape[0])
        or (arrValues.shape[0] != arrAziSample.shape[0])
        or (arrValues.shape[0] != arrCoEleSample.shape[0])
    ):
        raise ValueError(
            (
                "interpolateDataSphere:"
                + "Input arrays of sizes %d ele, %d azi, %d values dont match"
            )
            % (
                arrCoEleSample.shape[0],
                arrAziSample.shape[0],
                arrValues.shape[0],
            )
        )
    if arrAziInter.shape[0] != arrCoEleInter.shape[0]:
        raise ValueError(
            (
                "interpolateDataSphere:"
                + "Output arrays of sizes %d ele, %d azi dont match"
            )
            % (arrCoEleInter.shape[0], arrAziInter.shape[0])
        )
    if method == "SH":
        numN = options.get("numN", 6)
        if numN <= 0:
            raise ValueError(
                "interpolateDataSphere:"
                + "_genSHMatrix: numN must be greater than 0."
            )
        else:
            return _interpolateSH(
                arrCoEleSample,
                arrAziSample,
                arrValues,
                arrCoEleInter,
                arrAziInter,
                numN,
            )
    else:
        raise NotImplementedError(
            "interpolateDataSphere: Method not implemented."
        )


def _interpolateSH(
    arrCoEleSample: np.ndarray,
    arrAziSample: np.ndarray,
    arrValues: np.ndarray,
    arrCoEleInter: np.ndarray,
    arrAziInter: np.ndarray,
    numN: int,
) -> np.ndarray:
    """Interpolate function on Sphere using Spherical Harmonics

    See Dissertation of delGaldo for details.

    Parameters
    ----------
    arrCoEleSample : np.ndarray
        Sampled Co-Elevation positions in radians
    arrAziSample : np.ndarray
        Sampled Azimuth positions in radians
    arrValues : np.ndarray
        Sampled values
    arrCoEleInter : np.ndarray
        CoElevation positions we want the function to be evaluated in radians
    arrAziInter : np.ndarray
        Azimuth Positions we want the function to be evaluated in radians
    numN : int
        Order of the SH

    Returns
    -------
    np.ndarray
        Description of returned object.

    """

    # number of sampling points of the function
    numSamples = arrAziSample.shape[0]

    # matrix containing the basis functions evaluated at the
    # sampling positions. this one is used during the fitting of
    # the interpolation coefficients:
    # min || matSample * X - arrValues||_2^2
    matSample = _genSHMatrix(arrCoEleSample, arrAziSample, numN)

    # this matrix is used to generate the interpolated values, so it
    # contains the basis functions evaluated at the interpolation
    # points and we use the least squares fit to get the right linear
    # combinations
    matInter = _genSHMatrix(arrCoEleInter, arrAziInter, numN)

    # preserve the shape of the original data
    tplOrigShape = arrValues.shape

    # do the least squares fit
    arrLstSq = np.linalg.lstsq(
        matSample, arrValues.reshape((numSamples, -1)), rcond=-1
    )

    # extract the coefficients from the least squares fit
    arrCoeffs = arrLstSq[0]

    # calculate the interpolated values and return the same shape as
    # the input, but with different size in the interpolated first coordinate
    arrRes = matInter.dot(arrCoeffs).reshape((-1, *tplOrigShape[1:]))

    return arrRes


def _genSHMatrix(
    arrCoEle: np.ndarray, arrAzi: np.ndarray, numN: int
) -> np.ndarray:
    """Create a Matrix containing sampled Spherical harmonics

    Parameters
    ----------
    arrCoEle : np.ndarray
        CoElevation angles to evaluate at in radians
    arrAzi : np.ndarray
        Azimuth angles to evaluate at in radians
    numN : int
        Order of the SH basis > 0

    Returns
    -------
    np.ndarray
        Matrix containing sampled SH as its columns
    """

    # the spherical harmonics are always complex except in trivial
    # cases
    matR = np.zeros(
        (arrCoEle.shape[0], numN * (numN + 1) + 1), dtype="complex128"
    )

    # count the current basis element we are in
    numInd = 0

    # SH have two indices Y_LM with |L| <= M, see the scipy docu
    # on them
    for ii1 in range(numN + 1):
        for ii2 in range(ii1 + 1):
            # except for ii1 == 0, we always can generate two basis elements
            matR[:, numInd] = np.asarray(sph_harm(ii2, ii1, arrAzi, arrCoEle))
            if ii1 > 0:
                # note that matR[:, -numInd] = matR[:, numInd].conj()
                # would also work
                matR[:, -numInd] = np.asarray(
                    sph_harm(-ii2, ii1, arrAzi, arrCoEle)
                )
            numInd += 1
    return matR
