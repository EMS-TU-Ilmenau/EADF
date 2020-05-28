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
Auxiliary Methods
-----------------

These functions can be used in various places to make a lot of things
easier.
"""
import numpy as np

__all__ = [
    "cartesianToSpherical",
    "columnwiseKron",
    "sampleAngles",
    "toGrid",
]


def cartesianToSpherical(arrA: np.ndarray) -> np.ndarray:
    """Convert from 3D to 3D Spherical Coordinates

    This function calculates 3D Spherical Coordinates from 3D cartesian
    coordinates, where we asume the points are aligned along the first
    axis=0.

    Parameters
    ----------
    arrA : np.ndarray
        N x 3 input array of N x X x Y x Z values. must not be complex.

    Returns
    -------
    np.ndarray
        N x 3 array of N x Co-Ele (rad) x Azi (rad) x Norm values .

    """

    if arrA.shape[1] != 3:
        raise ValueError(
            "cartesianToSpherical: arrA has wrong second dimension."
        )

    if arrA.dtype in ["complex64", "complex128"]:
        raise ValueError("cartesianToSpherical: arrA is complex.")

    arrRes = np.empty((arrA.shape[0], 3), dtype="float")

    # Norm
    arrRes[:, 2] = np.linalg.norm(arrA, axis=1)

    # Co-Elevation
    arrRes[:, 0] = np.arccos(arrA[:, 2] / arrRes[:, 2])

    # Azimuth
    arrRes[:, 1] = np.arctan2(arrA[:, 1], arrA[:, 0])

    return arrRes


def columnwiseKron(arrA: np.ndarray, arrB: np.ndarray) -> np.ndarray:
    """Calculate column-wise Kronecker-Product

    Parameters
    ----------
    arrA : np.ndarray
        First input `arrA`.
    arrB : np.ndarray
        Second input `arrB`.

    Returns
    -------
    np.ndarray
        columnwisekron(arrA, arrB)

    """

    if arrA.shape[1] != arrB.shape[1]:
        raise ValueError("columnwiseKron: Matrices cannot be multiplied")

    # the first matrix needs its rows repeated as many times as the
    # other one has rows. the second one needs to be placed repeated
    # as a whole so many times as the first one has rows.
    # the we just do an elementwise multiplication and are done.
    return np.multiply(
        np.repeat(arrA, arrB.shape[0], axis=0),
        np.tile(arrB, (arrA.shape[0], 1)),
    )


def sampleAngles(numCoEle: int, numAzi: int, **kwargs) -> tuple:
    """Generate regular samplings in co-elevation and azimuth

    By default we generate angles in *co-elevation* and azimuth. This is due
    to the fact that the :py:obj:`EADF` works best in this case. Both
    directions are sampled regularly.

    Parameters
    ----------
    numCoEle : int
        Number of samples in co-elevation direction. > 0
    numAzi : int
        Number of samples in azimuth direction. > 0
    lstEndPoints : [0, 0], optional
        If endpoints should be generated in the respective dimensions

    Returns
    -------
    np.ndarray
        (2 x angles) in radians

    """
    if numAzi < 1:
        raise ValueError("sampleAngles: numAzi is %d, must be > 0" % (numAzi))
    if numCoEle < 1:
        raise ValueError(
            "sampleAngles: numCoEle is %d, must be > 0" % (numCoEle)
        )

    lstEndPoints = kwargs.get("lstEndPoints", [False, False])

    if len(lstEndPoints) != 2:
        raise ValueError(
            "sampleAngles: lstEndPoints has length %d instead of 2."
            % (len(lstEndPoints))
        )

    arrCoEle = np.linspace(0, +np.pi, numCoEle, endpoint=lstEndPoints[0])
    arrAzi = np.linspace(0, 2 * np.pi, numAzi, endpoint=lstEndPoints[1])
    return (arrCoEle, arrAzi)


def toGrid(*args) -> tuple:
    """Build up all pairwise combinations of angles

    For two given arrays of possibly unequal lengths N1, ..., NK we generate
    two new arrays, that contain all N1 x ... x NK pairwise combinations
    of the two array elements.

    Parameters
    ----------
    *args :
        several array like structures that make up the
        coordinate axes' grid points.

    Returns
    -------
    tuple
        contains K np.ndarrays

    """

    grdTpl = np.meshgrid(*args)
    return (tt.flatten() for tt in grdTpl)
