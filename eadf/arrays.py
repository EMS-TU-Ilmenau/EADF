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
Routines to Create Synthetic Arrays
-----------------------------------

Here we provide a convenient way to generate :py:obj:`EADF` objects from
synthetic arrays. We assume the arrays elements to be uniformly sensitive
in all directions.

Making your Own
^^^^^^^^^^^^^^^

If you want to add your of synthetic configurations, this is the place to
be. We suggest to use the generateArbitraryDipole function by just passing
the locations of the elements to it and let it handle the rest.

The Functions
^^^^^^^^^^^^^
"""

import numpy as np
from scipy.spatial.distance import cdist
import scipy.constants

from .eadf import EADF
from .auxiliary import toGrid
from .auxiliary import sampleAngles

__all__ = [
    "generateURA",
    "generateULA",
    "generateUCA",
    "generateStackedUCA",
    "generateArbitraryDipole",
    "generateArbitraryPatch",
]


def generateURA(
    numElementsY: int,
    numElementsZ: int,
    numSpacingY: float,
    numSpacingZ: float,
    elementSize: np.ndarray,
    arrFreq: np.ndarray,
    addCrossPolPort: bool = False,
    **options,
) -> EADF:
    """Uniform Rectangular Array in y-z-plane

    Creates an URA in y-z-plane according to the given parameters. Depending on
    the dimension of elementSize the basic elements are dipoles or patches. For
    dipoles the elementSize has to be (1 x 1). For patches the elementSize
    requires a shape of (3 x 1).

    The raw dipoles are located in the z-axis and the predominant polarization
    is vertical.
    The raw patches are located in the y-z-plane and the predominant
    polarization is horizontal.

    By setting the addCrosspolPort true, each element will be duplicated with
    a crosspol element. So the number of elements in the EADF is also doubled.
    The cross-pol element is rotated about 90 degree around the x-axis to get
    the cross-polarization.

    Example
    -------

    >>> import eadf
    >>> import numpy as np
    >>> elementSize = np.ones((3, 1))
    >>> arrFreq = np.arange(1,4)
    >>> A = eadf.generateURA(7, 5, 1.5, 0.5, elementSize, arrFreq)

    Parameters
    ----------
    numElementsY : int
        number of array elements in x-direction
    numElementsZ : int
        number of array elements in y-direction
    numSpacingY : float
        spacing between the first and last element in meter
    numSpacingZ : float
        spacing between the first and last element in meter
    elementSize : np.ndarray
        (dim x 1) array with size of single antenna element in meter
        (1 x 1) only length for dipole
        (3 x 1) length, width and thickness of patch element
    arrFreq : np.ndarray
        array of frequencies for EADF calculation in Hertz
    addCrossPolPort : bool
        Should we have appropriately rotated cross-pol ports? If true, the
        virtual elements will have two cross-polarized ports and are described
        as two elements in the EADF object. If false, the array has a
        predominant polarization.

        Defaults to false.
    **options:
        get passed to the EADF constructor

    Returns
    -------
    EADF
        URA
    """

    if numElementsY <= 0:
        raise ValueError("generateURA: numElementsY <= 0 is not allowed.")
    if numSpacingY <= 0:
        raise ValueError("generateURA: numSpacingY <= 0 is not allowed.")
    if numElementsZ <= 0:
        raise ValueError("generateURA: numElementsZ <= 0 is not allowed.")
    if numSpacingZ <= 0:
        raise ValueError("generateURA: numSpacingZ <= 0 is not allowed.")
    if np.any(elementSize <= 0):
        raise ValueError("generateURA: elementSize <= 0 is not allowed.")
    if np.any(arrFreq <= 0):
        raise ValueError("generateURA: frequency <= 0 is not allowed.")
    if not (elementSize.shape[0] == 1 or elementSize.shape[0] == 3):
        raise ValueError(
            "generateURA: elementSize has to be a 1- or 3-element array."
        )

    arrY = np.linspace(-numSpacingY / 2, +numSpacingY / 2, numElementsY)
    arrZ = np.linspace(-numSpacingZ / 2, +numSpacingZ / 2, numElementsZ)

    grdY, grdZ = toGrid(arrY, arrZ)

    # we align the elements along the x-coordinate
    arrPos = np.stack(
        (np.zeros(numElementsY * numElementsZ), grdY, grdZ), axis=0
    )

    arrRot = np.zeros((3, numElementsY * numElementsZ))
    arrSize = np.tile(elementSize, (1, numElementsY * numElementsZ))

    if elementSize.shape[0] == 1:
        return generateArbitraryDipole(
            arrPos,
            arrRot,
            arrSize,
            arrFreq,
            addCrossPolPort=addCrossPolPort,
            **options,
        )
    else:
        return generateArbitraryPatch(
            arrPos,
            arrRot,
            arrSize,
            arrFreq,
            addCrossPolPort=addCrossPolPort,
            **options,
        )


def generateULA(
    numElements: int,
    numSpacing: float,
    elementSize: np.ndarray,
    arrFreq: np.ndarray,
    addCrossPolPort: bool = False,
    **options,
) -> EADF:
    """Uniform Linear Array (ULA) along y-axis

    Creates an ULA along y-axis according to the given parameters. Depending on
    the dimension of elementSize the basic elements are dipoles or patches. For
    dipoles the elementSize has to be (1 x 1). For patches the elementSize
    requires a shape of (3 x 1).

    The raw dipoles are located in the z-axis and the predominant polarization
    is vertical.
    The raw patches are located in the y-z-plane and the predominant
    polarization is horizontal.

    By setting the addCrosspolPort true, each element will be duplicated with
    a crosspol element. So the number of elements in the EADF is also doubled.
    The cross-pol element is rotated about 90 degree around the x-axis to get
    the cross-polarization.

    Example
    -------

    >>> import eadf
    >>> import numpy as np
    >>> elementSize = np.ones((3, 1))
    >>> arrFreq = np.arange(1,4)
    >>> A = eadf.generateULA(11, 1.5, elementSize, arrFreq)

    Parameters
    ----------
    numElements : int
        number of array elements
    numSpacing : float
        spacing between the first and last element in meter
    elementSize : np.ndarray
        (dim x 1) array with size of single antenna element in meter
        (1 x 1) only length for dipole
        (3 x 1) length, width and thickness of patch element
    arrFreq : np.ndarray
        array of frequencies for EADF calculation in Hertz
    addCrossPolPort : bool
        Should we have appropriately rotated cross-pol ports? If true, the
        virtual elements will have two cross-polarized ports and are described
        as two elements in the EADF object. If false, the array has a
        predominant polarization.

        Defaults to false.

    Returns
    -------
    EADF
        ULA
    """

    if numElements <= 0:
        raise ValueError("generateULA: numElements <= 0 is not allowed.")
    if numSpacing <= 0:
        raise ValueError("generateULA: numSpacing <= 0 is not allowed.")
    if np.any(elementSize <= 0):
        raise ValueError("generateULA: elementSize <= 0 is not allowed.")
    if np.any(arrFreq <= 0):
        raise ValueError("generateULA: frequency <= 0 is not allowed.")
    if not (elementSize.shape[0] == 1 or elementSize.shape[0] == 3):
        raise ValueError(
            "generateULA: elementSize has to be a 1- or 3-element array."
        )

    # we align the elements along the x-coordinate
    arrPos = np.stack(
        (
            np.zeros(numElements),
            np.linspace(-numSpacing / 2, +numSpacing / 2, numElements),
            np.zeros(numElements),
        ),
        axis=0,
    )

    arrRot = np.zeros((3, numElements))
    arrSize = np.tile(elementSize, (1, numElements))

    if elementSize.shape[0] == 1:
        return generateArbitraryDipole(
            arrPos,
            arrRot,
            arrSize,
            arrFreq,
            addCrossPolPort=addCrossPolPort,
            **options,
        )
    else:
        return generateArbitraryPatch(
            arrPos,
            arrRot,
            arrSize,
            arrFreq,
            addCrossPolPort=addCrossPolPort,
            **options,
        )


def generateUCA(
    numElements: int,
    numRadius: float,
    elementSize: np.ndarray,
    arrFreq: np.ndarray,
    addCrossPolPort: bool = False,
    **options,
) -> EADF:
    """Uniform Circular Array (UCA) in the x-y-plane

    Creates an UCA in the x-y-plane according to the given parameters. Depending
    on the dimension of elementSize the basic elements are dipoles or patches.
    For dipoles the elementSize has to be (1 x 1). For patches the elementSize
    requires a shape of (3 x 1).

    The raw dipoles are located in the z-axis and the predominant polarization
    is vertical.
    The raw patches are located in the y-z-plane and the predominant
    polarization is horizontal.

    By setting the addCrosspolPort true, each element will be duplicated with
    a crosspol element. So the number of elements in the EADF is also doubled.
    The cross-pol element is rotated about 90 degree around the x-axis to get
    the cross-polarization.

    Example
    -------

    >>> import eadf
    >>> import numpy as np
    >>> elementSize = np.ones((3, 1))
    >>> arrFreq = np.arange(1,4)
    >>> A = eadf.generateUCA(11, 1.5, elementSize, arrFreq)

    Parameters
    ----------
    numElements : int
        Number of Elements
    numRadius : float
        Radius of the UCA in meter
    elementSize : np.ndarray
        (dim x 1) array with size of single antenna element in meter
        (1 x 1) only length for dipole
        (3 x 1) length, width and thickness of patch element
    arrFreq : np.ndarray
        array of frequencies for EADF calculation in Hertz
    addCrossPolPort : bool
        Should we have appropriately rotated cross-pol ports? If true, the
        virtual elements will have two cross-polarized ports and are described
        as two elements in the EADF object. If false, the array has a
        predominant polarization.

        Defaults to false.

    Returns
    -------
    EADF
        EADF object

    """
    if numElements <= 0:
        raise ValueError("generateUCA: numElements <= 0 is not allowed.")
    if numRadius <= 0:
        raise ValueError("generateUCA: numRadius <= 0 is not allowed.")
    if np.any(elementSize <= 0):
        raise ValueError("generateUCA: elementSize <= 0 is not allowed.")
    if np.any(arrFreq <= 0):
        raise ValueError("generateUCA: frequency <= 0 is not allowed.")
    if not (elementSize.shape[0] == 1 or elementSize.shape[0] == 3):
        raise ValueError(
            "generateUCA: elementSize has to be a 1- or 3-element array."
        )

    # create regular grid of angle positions for the elements
    arrElemAngle = np.linspace(0, 2 * np.pi, numElements, endpoint=False)

    # calculate the positions of the elements in 2D cartesian coordinates
    arrPos = np.stack(
        (
            numRadius * np.cos(arrElemAngle),
            numRadius * np.sin(arrElemAngle),
            np.zeros(numElements),
        ),
        axis=0,
    )

    # rotation of elements only arround z-axis
    arrRot = np.stack(
        (np.zeros(numElements), np.zeros(numElements), arrElemAngle), axis=0
    )
    # each element has the same size
    arrSize = np.tile(elementSize, (1, numElements))

    # call the routine for the arbitrary arrays
    if elementSize.shape[0] == 1:
        return generateArbitraryDipole(
            arrPos,
            arrRot,
            arrSize,
            arrFreq,
            addCrossPolPort=addCrossPolPort,
            **options,
        )
    else:
        return generateArbitraryPatch(
            arrPos,
            arrRot,
            arrSize,
            arrFreq,
            addCrossPolPort=addCrossPolPort,
            **options,
        )


def generateStackedUCA(
    numElements: int,
    numStacks: int,
    numRadius: float,
    numHeight: float,
    elementSize: np.ndarray,
    arrFreq: np.ndarray,
    addCrossPolPort: bool = False,
    **options,
) -> EADF:
    """Stacked Uniform Circular Array (SUCA)

    Creates a SUCA according to the given parameters. Depending on the dimension
    of elementSize the basic elements are dipoles or patches. For dipoles the
    elementSize has to be (1 x 1). For patches the elementSize requires a shape
    of (3 x 1).

    The raw dipoles are located in the z-axis and the predominant polarization
    is vertical.
    The raw patches are located in the y-z-plane and the predominant
    polarization is horizontal.

    By setting the addCrosspolPort true, each element will be duplicated with
    a crosspol element. So the number of elements in the EADF is also doubled.
    The cross-pol element is rotated about 90 degree around the x-axis to get
    the cross-polarization.

    Example
    -------

    >>> import eadf
    >>> import numpy as np
    >>> elementSize = np.ones((3, 1))
    >>> arrFreq = np.arange(1,4)
    >>> A = eadf.generateSteackedUCA(11, 3, 1.5, 0.5, elementSize, arrFreq)

    Parameters
    ----------
    numElements : int
        Number of Elements per Stack > 0
    numStacks : int
        Number of Stacks > 0
    numRadius : float
        Radius of the SUCA in meter
    numHeight : float
        Displacement height between two adjacent stacks in meter
    elementSize : np.ndarray
        (dim x 1) array with size of single antenna element in meter
        (1 x 1) only length for dipole
        (3 x 1) length, width and thickness of patch element
    arrFreq : np.ndarray
        array of frequencies for EADF calculation in Hertz
    addCrossPolPort : bool
        Should we have appropriately rotated cross-pol ports? If true, the
        virtual elements will have two cross-polarized ports and are described
        as two elements in the EADF object. If false, the array has a
        predominant polarization.

        Defaults to false.

    Returns
    -------
    EADF
        EADF object representing this very array

    """
    if numElements <= 0:
        raise ValueError(
            "generateStackedUCA: numElements <= 0 is not allowed."
        )
    if numStacks <= 0:
        raise ValueError("generateStackedUCA: numStacks <= 0 is not allowed.")
    if numRadius <= 0:
        raise ValueError("generateStackedUCA: numRadius <= 0 is not allowed.")
    if numHeight <= 0:
        raise ValueError("generateStackedUCA: numHeight <= 0 is not allowed.")
    if np.any(elementSize <= 0):
        raise ValueError(
            "generateStackedUCA: elementSize <= 0 is not allowed."
        )
    if np.any(arrFreq <= 0):
        raise ValueError("generateStackedUCA: frequency <= 0 is not allowed.")
    if not (elementSize.shape[0] == 1 or elementSize.shape[0] == 3):
        raise ValueError(
            "generateStackedUCA: elementSize has to be a 1- or 3-element array."
        )

    # regular grid of angles
    arrElemAngle = np.linspace(0, 2 * np.pi, numElements, endpoint=False)

    # regular grid of heights
    arrHeights = np.linspace(
        0, numStacks * numHeight, numStacks, endpoint=False
    )

    # array of position of elements in 3D cartesian coordinates
    arrPos = np.empty((3, numElements * numStacks))

    arrPos[:2, :] = np.tile(
        np.stack(
            (
                numRadius * np.cos(arrElemAngle),
                numRadius * np.sin(arrElemAngle),
            ),
            axis=0,
        ),
        (1, numStacks),
    )

    # the heights repeat according to the number of elements and the
    # number of stacks
    arrPos[2, :] = np.repeat(arrHeights, numElements)

    # rotation of elements only arround z-axis
    arrRot = np.tile(
        np.stack(
            (np.zeros(numElements), np.zeros(numElements), arrElemAngle),
            axis=0,
        ),
        (1, numStacks),
    )

    # each element has the same size
    arrSize = np.tile(elementSize, (1, numElements * numStacks))

    if elementSize.shape[0] == 1:
        return generateArbitraryDipole(
            arrPos,
            arrRot,
            arrSize,
            arrFreq,
            addCrossPolPort=addCrossPolPort,
            **options,
        )
    else:
        return generateArbitraryPatch(
            arrPos,
            arrRot,
            arrSize,
            arrFreq,
            addCrossPolPort=addCrossPolPort,
            **options,
        )


def generateArbitraryDipole(
    arrPos: np.ndarray,
    arrRot: np.ndarray,
    arrLength: np.ndarray,
    arrFreq: np.ndarray,
    addCrossPolPort: bool = False,
    **options,
) -> EADF:
    """Arbitrary dipole array EADF

    One specifies a (3 x N) np.ndarray to specify the elements positions and
    rotations in 3D cartesian space. Furthermore, a (1 x N) np.ndarray specifies
    the dipole-lengths of the elements. The EADF can be created for multiple
    frequencies at one time. The elements themselves are assumed to be
    uniform emitters based on finite length dipoles. One can decide if only one
    single pol dipole is used or if two dipoles are combined to a
    dual-polarimetric antenna element. This function allows to create a vast
    amount of different antenna geometries for quick testing.

    Example
    -------

    >>> import eadf
    >>> import numpy as np
    >>> arrPos = np.random.uniform(-1, 1, (3, 10))
    >>> arrRot = np.zeros((3, 10))
    >>> arrLength = np.ones((1, 10))
    >>> arrFreq = np.arange(1,4)
    >>> A = eadf.generateArbitraryDipole(arrPos, arrRot, arrLength, arrFreq)

    Parameters
    ----------
    arrPos : np.ndarray
        (3 x numElements) array of positions in meter
    arrRot : np.ndarray
        (3 x numElements) array of rotations of the elements in radians
    arrLength : np.ndarray
        (1 x numElements) array of length of the individual dipole lengths in
        meter
    arrFreq : np.ndarray
        array of frequencies to sample in Hertz
    addCrossPolPort : bool
        Should we have appropriately rotated cross-pol ports? If true, the
        virtual elements will have two cross-polarized ports and are described
        as two elements in the EADF object. If false, the array has a
        predominant vertical polarization.

        Defaults to false.

    Returns
    -------
    EADF
        EADF object representing this very array

    """

    if arrPos.shape[0] != 3:
        raise ValueError(
            "generateArbitraryDipole: arrPos must have exactly 3 rows"
        )

    if arrRot.shape[0] != 3:
        raise ValueError(
            "generateArbitraryDipole: arrRot must have exactly 3 rows"
        )

    if arrLength.shape[0] != 1:
        raise ValueError(
            "generateArbitraryDipole: arrLength must be a rowvector (1 x N)"
        )

    # currently only RotZ is possible --> doesn't change anything for dipole
    if np.any(arrRot[:2, :]):
        raise ValueError(
            "generateArbitraryDipole: only rotation around z-axis"
        )

    # upper bound for the radius enclosing sphere
    numMaxDist = 0.5 * np.max(cdist(arrPos, arrPos))

    # Now we sample in Azimuth an CoElevation according to (C.2.1.1.)
    # in the dissertation of del galdo
    numL = int(np.ceil(2 * np.pi * numMaxDist) + 10)
    numAzi = 4 * numL
    numCoEle = 2 * numL + 1

    # now calc the anuglar grids
    arrCoEle, arrAzi = sampleAngles(
        numCoEle, numAzi, lstEndPoints=[True, True]
    )
    # grdAzi, grdCoEle = toGrid(arrAzi, arrCoEle)
    grdAzi, grdCoEle = np.meshgrid(arrAzi, arrCoEle)

    # apply z-rotation to azimuth angles and create correct tensor dimensions
    rotZ = arrRot[2, :]
    grdAzi = (
        grdAzi[:, :, np.newaxis, np.newaxis, np.newaxis]
        - rotZ[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    )
    grdCoEle = grdCoEle[:, :, np.newaxis, np.newaxis, np.newaxis]
    arrLength = arrLength[np.newaxis, np.newaxis, np.newaxis, :]

    # radio parameters
    arrLambda = (
        scipy.constants.c
        / np.atleast_2d(arrFreq).T[np.newaxis, np.newaxis, :, np.newaxis]
    )
    arrK = 2 * np.pi / arrLambda

    # compute beampattern with handling of division by zero
    ETheta = np.asarray(
        np.divide(
            (
                np.cos(arrK * arrLength / 2 * np.cos(grdCoEle))
                - np.cos(arrK * arrLength / 2)
            ),
            np.sin(grdCoEle),
            out=np.zeros_like(np.repeat(grdAzi, arrFreq.shape[0], axis=2)),
            where=np.sin(grdCoEle) != 0,
        )
    )
    EPhi = np.zeros(ETheta.shape)

    if addCrossPolPort:
        # rotate dipole 90 degree around x-axis to create second polarization
        xV = np.sin(grdCoEle) * np.cos(grdAzi)
        yV = np.sin(grdCoEle) * np.sin(grdAzi)
        zV = np.cos(grdCoEle)

        xH = xV
        yH = -zV
        zH = yV

        grdAziH = np.arctan2(yH, xH)
        grdCoEleH = np.arccos(zH)

        # compute beampattern with handling of division by zero
        EPhiH = np.asarray(
            np.divide(
                (
                    np.cos(arrK * arrLength / 2 * np.cos(grdCoEleH))
                    - np.cos(arrK * arrLength / 2)
                ),
                np.sin(grdCoEleH),
                out=np.zeros_like(
                    np.repeat(grdAziH, arrFreq.shape[0], axis=2)
                ),
                where=(np.sin(grdCoEleH) != 0),
            )
        )
        EThetaH = np.zeros(EPhiH.shape)

        # combine H- and V-port for addCrossPolPortarized antenna element
        EDualPol = np.concatenate(
            (
                np.concatenate((EPhiH, EThetaH), axis=3),
                np.concatenate((EPhi, ETheta), axis=3),
            ),
            axis=4,
        )
    else:
        EDualPol = np.concatenate((EPhi, ETheta), axis=3)

    # projections of the angles on the sphere
    arrProj = np.stack(
        (
            np.cos(grdAzi) * np.sin(grdCoEle),
            np.sin(grdAzi) * np.sin(grdCoEle),
            np.repeat(np.cos(grdCoEle), arrPos.shape[1], axis=4),
        ),
        axis=5,
    )

    # wavecevtors for the different angles
    k = arrK[:, :, :, :, :, np.newaxis] * arrProj

    # phases at each element according to wavevectors and element positions
    phases = np.exp(-1j * np.einsum("abcdef,fe->abcde", k, arrPos))

    # repeat array elements such that all H-ports will be first
    if addCrossPolPort:
        phases = np.tile(phases, (1, 1, 1, 1, 2))
        arrPos = np.repeat(arrPos, 2, axis=1)

    # multiply beampattern with the calculate phases for each element
    arrEADFData = EDualPol * phases

    return EADF(arrEADFData, arrCoEle, arrAzi, arrFreq, arrPos, **options,)


def generateArbitraryPatch(
    arrPos: np.ndarray,
    arrRot: np.ndarray,
    arrSize: np.ndarray,
    arrFreq: np.ndarray,
    addCrossPolPort: bool = False,
    **options,
) -> EADF:
    """Creates an EADF of an analyical patch antenna and returns an EADF.

    One specifies a (3 x N) np.ndarray to specify the elements positions and
    rotations in 3D cartesian space. Furthermore, a (3 x N) np.ndarray specifies
    the size of the patch elements. The EADF can be created for multiple
    frequencies at one time. One can decide if only one single pol patch is
    used or if two dipoles are combined to a dual-polarimetric antenna element.
    This function allows to create a vast amount of different antenna geometries
    for qucik testing.

    Example
    -------

    >>> import eadf
    >>> import numpy as np
    >>> arrPos = np.random.uniform(-1, 1, (3, 10))
    >>> arrRot = np.zeros((3, 10))
    >>> arrSize = np.ones((3, 10))
    >>> arrFreq = np.arange(1,4)
    >>> A = eadf.generateArbitraryPatch(arrPos, arrRot, arrSize, arrFreq)

    Parameters
    ----------
    arrPos : np.ndarray
        (3 x numElements) array of positions in meter
    arrRot : np.ndarray
        (3 x numElements) array of rotations of the elements in radians
    arrSize : np.ndarray
        (3 x numElements) array of sizes of the patches (length, width,
        thickness) in meter
    arrFreq : np.ndarray
        array of frequencies to sample in Hertz
    addCrossPolPort : bool
        Should we have appropriately rotated cross-pol ports? If true, the
        virtual elements will have two cross-polarized ports and are described
        as two elements in the EADF object. If false, the array has a
        predominant horizontal polarization.

        Defaults to false.

    Returns
    -------
    EADF
        EADF object

    """
    if arrPos.shape[0] != 3:
        raise ValueError(
            "generateArbitraryPatch: arrPos must have exactly 3 rows"
        )

    if arrRot.shape[0] != 3:
        raise ValueError(
            "generateArbitraryPatch: arrRot must have exactly 3 rows"
        )

    if arrSize.shape[0] != 3:
        raise ValueError(
            "generateArbitraryPatch: arrSize must have exactly 3 rows"
        )

    arrPos = np.asarray(arrPos)
    arrRot = np.asarray(arrRot)
    arrSize = np.asarray(arrSize)
    arrFreq = np.asarray(arrFreq)

    # currently only RotZ is possible
    if np.any(arrRot[:2, :]):
        raise ValueError("generateArbitraryPatch: only rotation around z-axis")

    # defining angle grid
    deltaAzi = 4 / 180 * np.pi
    deltaEle = 2 / 180 * np.pi

    arrCoEle, arrAzi = sampleAngles(
        int(np.pi / deltaEle),
        int(2 * np.pi / deltaAzi),
        lstEndPoints=[True, False],
    )
    gridAzi, gridCoEle = np.meshgrid(arrAzi, arrCoEle)

    # apply z-rotation to azimuth angles
    rotZ = arrRot[2, :]
    gridAzi = (
        gridAzi[:, :, np.newaxis, np.newaxis, np.newaxis]
        - rotZ[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    )
    gridCoEle = gridCoEle[:, :, np.newaxis, np.newaxis, np.newaxis]

    # size of patch
    arrWidth = arrSize[1, :][np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    arrLength = arrSize[0, :][
        np.newaxis, np.newaxis, np.newaxis, np.newaxis, :
    ]
    arrHeight = arrSize[2, :][
        np.newaxis, np.newaxis, np.newaxis, np.newaxis, :
    ]

    # radio parameters
    arrLambda = (
        scipy.constants.c
        / np.atleast_2d(arrFreq).T[np.newaxis, np.newaxis, :, np.newaxis]
    )
    arrK = 2 * np.pi / arrLambda

    # analytical description of Beampattern
    # source "Wiley - Antenna Theory" p. 743ff, equations (14-40) and (14-43)
    X = arrK * arrHeight / 2 * np.cos(gridCoEle)
    Z = arrK * arrWidth / 2 * np.sin(gridCoEle) * np.cos(gridAzi)

    # calculate field normalized field compontents, such that the maximum of
    # the beampattern will be 1 for the optimal wavelength / wavefactor
    EPhi = (
        1j
        * arrK
        * arrHeight
        * arrWidth
        / (2 * np.pi)
        * np.sin(gridCoEle)
        * np.sinc(X / np.pi)
        * np.sinc(Z / np.pi)
        * np.cos(arrK * arrLength / 2 * np.sin(gridCoEle) * np.sin(gridAzi))
    )
    ETheta = np.zeros_like(EPhi)

    if addCrossPolPort:
        # calculate angle grid for V port with rotated beampattern
        xH = np.sin(gridCoEle) * np.cos(gridAzi)
        yH = np.sin(gridCoEle) * np.sin(gridAzi)
        zH = np.cos(gridCoEle)

        xV = xH
        yV = -zH
        zV = yH

        gridAziV = np.arctan2(yV, xV)
        gridCoEleV = np.arccos(zV)

        XV = arrK * arrHeight / 2 * np.cos(gridCoEleV)
        ZV = arrK * arrWidth / 2 * np.sin(gridCoEleV) * np.cos(gridAziV)

        EPhiV = np.zeros(EPhi.shape)
        EThetaV = (
            1j
            * arrK
            * arrHeight
            * arrWidth
            / (2 * np.pi)
            * np.sin(gridCoEleV)
            * np.sinc(XV / np.pi)
            * np.sinc(ZV / np.pi)
            * np.cos(
                arrK * arrLength / 2 * np.sin(gridCoEleV) * np.sin(gridAziV)
            )
        )

        # combine H- and V-port for addCrossPolPortarized antenna element
        EDualPol = np.concatenate(
            (
                np.concatenate((EPhi, ETheta), axis=3),
                np.concatenate((EPhiV, EThetaV), axis=3),
            ),
            axis=4,
        )

    else:
        EDualPol = np.concatenate((EPhi, ETheta), axis=3)

    # wavevectors
    kx = np.cos(gridAzi) * np.sin(gridCoEle)
    ky = np.sin(gridAzi) * np.sin(gridCoEle)
    kz = np.cos(gridCoEle)
    k = arrK[:, :, :, :, :, np.newaxis] * np.stack(
        (kx, ky, np.repeat(kz, arrPos.shape[1], axis=4)), axis=5
    )

    # calculate phases at position of each array element
    phases = np.exp(-1j * np.einsum("abcdef,fe->abcde", k, arrPos))

    # repeat array elements such that all H-ports will be first
    if addCrossPolPort:
        phases = np.tile(phases, (1, 1, 1, 1, 2))
        arrPos = np.repeat(arrPos, 2, axis=1)

    BP = EDualPol * phases

    return EADF(BP, arrCoEle, arrAzi, arrFreq, arrPos, **options,)
