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
Importers
---------

Here we provide a collection of several importers to conveniently
create EADF objects from various data formats. For this purpose we
provide a set of so called *handshake formats*, which can be seen as
intermediate formats, which facilitate the construction of importers,
since for these handshake formats we already provide tested conversion
routines to the internal data format.

For these formats there are readily available and tested importers. See
the respective importer methods for further details.

 - Regular (in space) Angular Data: 2*co-ele x azi x pol x freq x elem.
    This format is simply handled by the EADF class initialization. So, if your
    data is already in that format, just call EADF() with it.
 - Regular (in space) Spatial Fourier Data:
   2*co-ele-sptfreq x azi-sptfreq x pol x freq x elem
 - Angle List Data: Ang x Pol x Freq x Elem
 - Regular (in space) Variation 2 Angular Data: ele x azi x elem x pol x freq.
 - Wideband Angular Data as struct from .mat file:
    This format accounts for array pattern information stored with a .mat file
    containing a Matlab structured array (struct) with 'Value' field having
    dimensions; ele x azi x elem x pol x freq.
 - Narrowband Angular Data as struct from .mat file:
    This format accounts for array pattern information stored with a .mat file
    containing a Matlab structured array (struct) with 'Value' field having
    dimensions; ele x azi x elem x pol.
 - HFSS export (.csv Files)
"""

import numpy as np
import scipy.io
import os.path
from warnings import warn
from itertools import product
import pandas as pd

from .core import ureg
from .eadf import EADF
from .sphericalharm import interpolateDataSphere
from .auxiliary import sampleAngles
from .auxiliary import toGrid
from .backend import rDtype

__all__ = [
    "fromAngleListData",
    "fromWidebandAngData",
    "fromNarrowBandAngData",
    "fromArraydefData",
    "fromHFSS",
]


def fromAngleListData(
    arrCoEleData: np.ndarray,
    arrAziData: np.ndarray,
    arrAngleListData: np.ndarray,
    arrFreqData: np.ndarray,
    arrPos: np.ndarray,
    numCoEle: int,
    numAzi: int,
    numErrorTol=1e-4,
    method="SH",
    **eadfOptions
) -> EADF:
    """Importer from the Angle List Data Handshake format

    This format allows to specify a list of angles (ele, azi)_i and
    beam pattern values v_i = (pol, freq, elem)_i which are then
    interpolated along the two angular domains to get a regular grid in
    azimuth and co-elevation. By default this is done using vector spherical
    harmonics, since they can deal with irregular sampling patterns
    quite nicely. In this format for each angular sampling point, we
    need to have excited the array elements with the same frequencies.

    Parameters
    ----------
    arrCoEleData : np.ndarray
        Sampled Co-elevation Angles in radians
    arrAziData : np.ndarray
        Sampled Azimuth Angles in radians
    arrAngleListData : np.ndarray
        List in  Angle x Freq x Pol x Element format
    arrFreqData : np.ndarray
        Frequencies the array was excited with in ascending order
    arrPos : np.ndarray
        Positions of the array elements
    numCoEle : int
        number of regular elevation samples used during interpolation > 0
    numAzi : int
        number of regular azimuth samples used during interpolation > 0
    numErrorTol : float
        error tolerance for coefficients fitting > 0
    method : string
        Interpolation Method, default='SH'
    eadfOptions:
        Things to tell the EADF constructor

    Returns
    -------
    EADF
        Created Array

    """
    if (
        (arrAziData.shape[0] != arrCoEleData.shape[0])
        or (arrAngleListData.shape[0] != arrAziData.shape[0])
        or (arrAngleListData.shape[0] != arrCoEleData.shape[0])
    ):
        raise ValueError(
            (
                "fromAngleListData: Input arrays"
                + " of sizes %d ele, %d azi, %d values dont match"
            )
            % (
                arrCoEleData.shape[0],
                arrAziData.shape[0],
                arrAngleListData.shape[0],
            )
        )
    if arrPos.shape[1] != arrAngleListData.shape[3]:
        raise ValueError(
            (
                "fromAngleListData:"
                + "Number of positions %d does not match provided data %d"
            )
            % (arrPos.shape[1], arrAngleListData.shape[3])
        )
    if arrFreqData.shape[0] != arrAngleListData.shape[1]:
        raise ValueError(
            (
                "fromAngleListData:"
                + "Number of freqs %d does not match provided data %d"
            )
            % (arrFreqData.shape[0], arrAngleListData.shape[1])
        )
    if numAzi < 0:
        raise ValueError("fromAngleListData: numAzi must be larger than 0.")
    if numCoEle < 0:
        raise ValueError("fromAngleListData: numCoEle must be larger than 0.")

    # we start with SH order of 5, see below
    # as soon as we offer more interpolation methods, we should handle
    # this differently
    numInterError = np.inf
    numN = 4

    # we steadily increase the approximation base size
    while numInterError > numErrorTol:
        numN += 1
        arrInter = interpolateDataSphere(
            arrCoEleData,
            arrAziData,
            arrAngleListData,
            arrCoEleData,
            arrAziData,
            numN=numN,
            method=method,
        )

        # calculate the current interpolation error
        numInterError = np.linalg.norm(
            arrInter - arrAngleListData
        ) / np.linalg.norm(arrAngleListData)

    # generate the regular grid, where we want to sample the array
    arrCoEle, arrAzi = sampleAngles(
        numCoEle, numAzi, lstEndPoints=[True, True]
    )
    grdCoEle, grdAng = toGrid(arrCoEle, arrAzi)

    # now run the interpolation for the regular grid, flip the pattern
    arrInter = interpolateDataSphere(
        arrCoEleData,
        arrAziData,
        arrAngleListData,
        grdCoEle,
        grdAng,
        numN=numN,
        method=method,
    ).reshape(
        numCoEle,
        numAzi,
        arrFreqData.shape[0],
        arrAngleListData.shape[1],
        arrPos.shape[1],
    )

    return EADF(arrInter, arrCoEle, arrAzi, arrFreqData, arrPos, **eadfOptions)


def fromWidebandAngData(path: str, **kwargs) -> EADF:
    """
    This format defines angular data over uniformly sampled
    ele and azi for a range of antenna ports elem, with v and/or h-polarization
    pol for frequency range freq. Using the importer it is possible to choose
    the respective indices for antenna ports, polarization and frequency
    to derive the EADF.

    The .mat file which utilizes this importer is expected to consist of a
    struct named 'pattern' which includes the fields:
    'Dim', 'Value', 'Description', 'Unit', and 'Date'.

    The 'Dim' field is itself a struct which describes the dimensions of the
    5-dimensional 'Value' tensor. 'Dim' contains Name-Value pairs corresponding
    to and ordered as 'Elevation', 'Azimuth', 'Element', 'Polarization',
    and 'Frequency'.

    The 'Elevation' and 'Azimuth' fields contain the vectors with the
    range of their respective angles.The 'Element' field contains the indices
    of the measured array ports starting from 1, while the 'Polarization' field
    is a cell array structured as {'h'} and/or {'v'}. The 'Frequency' field
    contains an array of the available frequency points.

    The 'Value' field is a tensor containing complex-double type data and
    is of the size (numEle x numAzi x numElem x numPol x numFreq).

    The 'Description' field contains a string describing what data is included
    in the file. Typically, it is 'sqrt of abs of complex realised gain'.

    The 'Unit' field is a string denoting if the data is in the linear
    or logarithmic units. Typically, it is expected to be 'linear'.

    The 'Date' field is a string denoting the date and time when the
    measurement campaign was completed and the data was stored.

    Parameters
    ----------
    path : string
        Directory location and name of .mat file containing measurement data
    arrPorInd : numpy.array, default = [], optional
        Indices of antenna ports
    arrPol : numpy.array, default = ['h','v'], optional
        Strings corresponding to polarization ('h' and/or 'v')
    arrFreq : numpy.array, default = 0, optional
        Frequency points in Hertz

    Returns
    -------
    EADF
        Created Array

    """

    if os.path.isfile(path):
        data = scipy.io.loadmat(path)
    else:
        raise IOError("fromWidebandAngData: file %s does not exist." % (path,))

    if not ("pattern" in data.keys()):
        raise KeyError("fromWidebandAngData: key 'pattern' does not exist.")

    arrCoEle = (
        np.radians(data["pattern"][0][0][0][0][0][1].flatten().astype(rDtype))
        + np.pi / 2
    )
    arrCoEle = np.linspace(arrCoEle[0], arrCoEle[-1], arrCoEle.shape[0])
    arrAzi = np.radians(
        data["pattern"][0][0][0][0][1][1].flatten().astype(rDtype)
    )
    arrAzi = np.linspace(arrAzi[0], arrAzi[-1], arrAzi.shape[0])
    arrFreq = data["pattern"][0][0][0][0][4][1]
    arrPattern = data["pattern"][0][0][1]
    # bring the data in the EADF data format
    arrPattern = np.swapaxes(arrPattern, 2, 4)

    arrPorInd = data["pattern"][0][0][0][0][2][1] - 1
    arrPol = data["pattern"][0][0][0][0][3][1]

    arrPorIndUser = kwargs.get("arrPorInd", [])
    arrPolUser = kwargs.get("arrPol", ["h", "v"])
    arrFreqUser = kwargs.get("arrFreq", 0)

    if arrPorIndUser in arrPorInd:
        arrPorInd = arrPorIndUser

    numPor = len(arrPorInd) or arrPorInd.size
    arrPos = np.random.randn(3, numPor)

    if arrFreqUser in arrFreq:
        arrFreqInd = np.where(arrFreq == arrFreqUser)[0]
        arrFreq = arrFreqUser
    else:
        arrFreqInd = np.array(range(0, arrFreq.size))
    if arrPolUser in arrPol:
        arrPolInd = np.where(arrPol == arrPolUser)[1]
    else:
        arrPolInd = np.array(range(0, arrPol.size))

    arrPatternEffective = arrPattern[:, :, arrFreqInd, :, :][
        :, :, :, arrPolInd, :
    ][:, :, :, :, arrPorInd][..., 0]

    return EADF(arrPatternEffective, arrCoEle, arrAzi, arrFreq, arrPos)


def fromNarrowBandAngData(path: str, **kwargs) -> EADF:
    """
    This format defines angular data over uniformly sampled
    ele and azi for a range of antenna ports elem, with v and/or h-polarization
    pol. Using the importer it is possible to choose
    the respective indices for antenna ports, polarization to derive the EADF.

    The .mat file which utilizes this importer is expected to consist of a
    struct named 'pattern' which includes the fields:
    'Dim', 'Value'.

    The 'Dim' field is itself a struct which describes the dimensions of the
    4-dimensional 'Value' tensor. 'Dim' contains Name-Value pairs corresponding
    to and ordered as 'Elevation', 'Azimuth', 'Element', and 'Polarization'.

    The 'Elevation' and 'Azimuth' fields contain the vectors with the
    range of their respective angles.The 'Element' field contains the indices
    of the measured array ports starting from 1, while the 'Polarization' field
    is a cell array structured as {'h'} and/or {'v'}.

    The 'Value' field is a tensor containing complex-double type data and
    is of the size (numEle x numAzi x numElem x numPol).

    Parameters
    ----------
    path : string
        Directory location and name of .mat file containing measurement data
    arrPorInd : numpy.array, default = [], optional
        Indices of antenna ports
    arrPol : numpy.array, default = ['h','v'], optional
        Strings corresponding to polarization ('h' and/or 'v')

    Returns
    -------
    EADF
        Created Array

    """

    if os.path.isfile(path):
        data = scipy.io.loadmat(path)
    else:
        raise IOError(
            "fromNarrowBandAngData: file %s does not exist." % (path,)
        )

    if not ("pattern" in data.keys()):
        raise KeyError("fromNarrowBandAngData: key 'pattern' does not exist.")

    arrCoEle = np.flip(
        np.radians(data["pattern"][0][0][1][0][0][1].flatten()) + np.pi / 2
    )

    arrAzi = np.radians(data["pattern"][0][0][1][0][1][1].flatten())
    arrPorInd = data["pattern"][0][0][1][0][2][1]
    arrPol = data["pattern"][0][0][1][0][3][1]
    arrPattern = data["pattern"][0][0][0]

    # fixed value for frequency
    numFreq = np.array([1])

    # add dimension for frequency
    arrPattern = arrPattern[:, :, :, :, np.newaxis]

    # flip pattern along coelevation to bring it in the EADF fromat
    arrPattern = np.flip(arrPattern, 0)

    # bring the data in the EADF data format
    arrPattern = np.swapaxes(arrPattern, 2, 4)

    arrPorIndUser = kwargs.get("arrPorIndUser", [])
    arrPolUser = kwargs.get("arrPolUser", ["h", "v"])

    if arrPorIndUser in arrPorInd:
        arrPorInd = arrPorIndUser

    numPor = len(arrPorInd) or arrPorInd.size
    arrPos = np.random.randn(3, numPor)

    if arrPolUser in arrPol:
        arrPolInd = np.where(arrPol == arrPolUser)[1]
    else:
        arrPolInd = np.array(range(0, arrPol.size))

    arrPatternEffective = arrPattern[:, :, :, arrPolInd, :][
        :, :, :, :, arrPorInd
    ]

    if arrPatternEffective.ndim > 5:
        arrPatternEffective = arrPatternEffective[:, :, :, :, :, 0]

    return EADF(arrPatternEffective, arrCoEle, arrAzi, numFreq, arrPos)


def fromArraydefData(path: str) -> EADF:
    """Import from Matlab arraydef structure used for calibrated arrays at TUI.

    Parameters
    ----------
    path : str
        path to .mat-file

    Returns
    -------
    EADF
        EADF object from the respective data

    """

    if os.path.isfile(path):
        data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    else:
        raise IOError("fromArraydefData: file %s does not exist." % (path,))

    if not ("arraydef" in data.keys()):
        raise KeyError("fromArraydefData: key 'arraydef' does not exist.")

    apertur = data["arraydef"].apertur

    # get number of samples for azi und ele (with redundancy; without center)
    numAzi = (apertur.saz - 1) * 2
    numEle = (apertur.sele - 1) * 2
    # get number of elements and so on
    numPort = apertur.elements
    numPol = apertur.pol
    numFreq = 1

    # take flipped parts of EADF
    G13 = apertur.G13
    G24 = apertur.G24

    # matrix for correct addition of the different blocks
    M = (
        np.array(
            [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]
        )
        / 4
    )
    # matrix for correct addition of the different center vectors
    m = np.array([[1, 1], [1, -1]]) / 2

    GkT = np.empty((numEle // 2, numAzi // 2, 4), dtype=np.complex128)
    G = np.empty(
        (numEle + 1, numAzi + 1, numFreq, numPol, numPort), dtype=np.complex128
    )
    # loop over elements and polarizations and create full EADF matrix
    for idx in range(numPort):
        for pol in range(numPol):

            Gk1 = G13[
                :,
                (pol * numPort + idx)
                * (numAzi // 4) : (pol * numPort + idx + 1)
                * (numAzi // 4),
            ]
            Gk3 = (
                G13[
                    :,
                    (numPol * numPort + pol * numPort + idx)
                    * (numAzi // 4) : (
                        numPol * numPort + pol * numPort + idx + 1
                    )
                    * (numAzi // 4),
                ]
                / 1j
            )
            Gk2 = (
                G24[
                    :,
                    (pol * numPort + idx)
                    * (numAzi // 4 + 1) : (pol * numPort + idx + 1)
                    * (numAzi // 4 + 1),
                ]
                / 1j
            )
            Gk4 = (
                G24[
                    :,
                    (numPol * numPort + pol * numPort + idx)
                    * (numAzi // 4 + 1) : (
                        numPol * numPort + pol * numPort + idx + 1
                    )
                    * (numAzi // 4 + 1),
                ]
                / -1
            )

            # interleave with blocks with zeros and write it to tensor
            GkT[:, 0::2, 0] = np.zeros((numEle // 2, numAzi // 4))
            GkT[:, 1::2, 0] = Gk1[:-1, :]
            GkT[:, 0::2, 1] = Gk2[:-1, :-1]
            GkT[:, 1::2, 1] = np.zeros((numEle // 2, numAzi // 4))
            GkT[:, 0::2, 2] = np.zeros((numEle // 2, numAzi // 4))
            GkT[:, 1::2, 2] = Gk3[:-1, :]
            GkT[:, 0::2, 3] = Gk4[:-1, :-1]
            GkT[:, 1::2, 3] = np.zeros((numEle // 2, numAzi // 4))

            # make multiplication with flipping matrix M for correct summation
            # of the single blocks
            # will return the original blocks of the big matrix, but some of
            # them are flipped, so we will flip them back before concatenation
            BT = np.einsum("ij,abj->abi", M, GkT, optimize="optimal")

            # take vectors where index of phi or theta is zero
            # for theta=0 vectors take the zerointerleaved ones
            G_phi_0 = np.stack((np.zeros(numEle // 2), Gk2[:-1, -1]), 1)
            G_theta_0 = np.vstack(
                (
                    np.reshape(
                        np.vstack((Gk2[-1, :-1], Gk1[-1, :])),
                        (1, -1),
                        order="F",
                    ),
                    np.reshape(
                        np.vstack((Gk4[-1, :-1], Gk3[-1, :])),
                        (1, -1),
                        order="F",
                    ),
                )
            )

            # make the multiplication with small flipping matrix m for correct
            # summation of the center vectors of the big matrix
            B_phi_0 = np.einsum("ij,aj->ai", m, G_phi_0)
            B_theta_0 = np.einsum("ij,ja->ia", m, G_theta_0)

            # concatenation of the flipped results to get the complete matrix
            G[:, :, 0, pol, idx] = np.block(
                [
                    [BT[:, :, 0], B_phi_0[:, [0]], np.fliplr(BT[:, :, 1])],
                    [
                        B_theta_0[[0], :],
                        np.array(Gk2[-1, -1]),
                        np.fliplr(B_theta_0[[1], :]),
                    ],
                    [
                        np.flipud(BT[:, :, 2]),
                        np.flipud(B_phi_0[:, [1]]),
                        np.flipud(np.fliplr(BT[:, :, 3])),
                    ],
                ]
            )

    # create angle lists
    arrAzi = np.linspace(0, 2 * np.pi, numAzi, endpoint=False)
    arrEle = np.fft.fftshift(
        np.linspace(0, 2 * np.pi, numEle, endpoint=False)
    )[: numEle // 2 + 1]

    muEle = np.arange(
        -np.floor((numEle + 1) // 2), np.floor((numEle + 1) // 2) + 1
    )
    muAzi = np.arange(
        -np.floor((numAzi + 1) // 2), np.floor((numAzi + 1) // 2) + 1
    )

    # create DFT matrices
    DEle = np.exp(1j * np.outer(arrEle, muEle))
    DAzi = np.exp(1j * np.outer(muAzi, np.fft.fftshift(arrAzi)))

    # compute IDFT
    BP = np.einsum("ia,ab...,bj->ij...", DEle, G, DAzi, optimize="optimal")

    # create angle vectors and so on
    arrCoEle = np.linspace(0, np.pi, numEle // 2 + 1)
    arrFreq = np.ones(1)
    if hasattr(data["arraydef"], "posData"):
        arrPos = data["arraydef"].PosData.T
    else:
        arrPos = np.ones((3, numPort))

    return EADF(BP, arrCoEle, arrAzi, arrFreq, arrPos)


# #############################################################################
# #  HFSS CSV Importer
# #############################################################################

# ##################################### Definitions
# Name constants: keynames for internal structures
NAME_UNIT = "unit"
NAME_TYPE = "type"
NAME_DATA = "data"
NAME_COL_NAME = "col_name"
NAME_COL_INDICES = "col_indices"
NAME_ROW_NAME = "row_name"
NAME_ROW_INDICES = "row_indices"

# Name constants: HFSS data representation identifiers
NAME_REAL = "re"
NAME_IMAGINARY = "im"
NAME_MAGNITUDE = "mag"
NAME_PHASE = "ang_rad"

# Type descriptors: Which type format is described by which data descriptors
DESC_REAL = set([None])
DESC_COMPLEX = set([NAME_REAL, NAME_IMAGINARY])
DESC_POLAR = set([NAME_MAGNITUDE, NAME_PHASE])
DESC_ALL = [DESC_REAL, DESC_COMPLEX, DESC_POLAR]


# ##################################### Helper functions
def parse_parantheses(expr):
    """
    Analyze a string containing nested pairs of parantheses (`()`).

    Parameters
    ----------
    expr : str
        A string containing nested parantheses.

    Returns
    -------
    list
        A list representation of the input string. Every matched pair of
        parantheses opens up another level of this return structure in a
        recursive manner. Multiple parantheses are supported and their
        order, corresponding to `expr`, is maintained in the output.
        String portions in before, in between or after matching parantheses
        are put as literal strings.
    """

    def _helper(tokens):
        items = []
        for item in tokens:
            if item == "(":
                result, closeparen = _helper(tokens)
                if not closeparen:
                    raise ValueError(
                        "bad expression -- unbalanced parentheses"
                    )
                items.append(result)
            elif item == ")":
                return items, True
            elif len(items) == 0 or isinstance(items[-1], list):
                items.append(item)
            else:
                items[-1] += item
        return items, False

    return _helper(iter(expr))[0]


def hfss_read_csv_to_dataframe(filename):
    """
    Read a HFSS .csv file and prepare it as pandas data frame.

    During the import, numbers will directly be interpreted as float and
    missing values are removed from the frame.
    Columns whose descriptor starts with `Unnamed` will be ignored.

    Parameters
    ----------
    filename : str
        Name of the .csv file to read

    Returns
    -------
    DataFrame
        The pandas DataFrame containing the data
    """
    frame = (
        pd.read_csv(
            filename,
            index_col=0,
            usecols=lambda name: not name.startswith("Unnamed"),
        )
        .dropna(thresh=2)
        .T.apply(pd.to_numeric, errors="ignore", downcast="float")
    )
    return frame


def hfss_interpret_header(item):
    """
    Decode parameter and type information from one (row or column) header item.

    Each header identifier contains multiple tokens, separated by spaces.
    Within one header string, only one unit identifier and only one data type
    identifier may be contained. If multiple of those will be found, a
    `ValueError` exception will be raised.

    A parameter is given as `key=value` token. A unit identifier is
    encapsulated in square brackets (`[]`) and the type identifier is a plain
    string not matching the other two formats. Only exactly one type or unit
    identifier is accepted in `item`.

    The function returns a dictionary, which contains all parameters as
    key-value pairs. The special identifier `unit` is used to carry the data
    unit found within the header and, accordingly, `type` corresponds to the
    data type.

    Parameters
    ----------
    item : str
        Header string in question

    Returns
    -------
    dict
        A dictionary containing all parameters, data types and unit identifiers
        found encoded in the given string.
    """

    def check_bracket(token, bracket):
        return len(token) >= 2 and (token[0] + token[-1]) == bracket

    tokens = item.split(" ")
    properties = {}
    for tt in tokens:
        if "=" in tt:
            # here's a property. Ignore it for now
            key, value = tt.split("=")
            if check_bracket(value, "''") or check_bracket(value, '""'):
                value = value[1:-1]
            properties[key.lower()] = ureg.Quantity(value)
        elif check_bracket(tt, bracket="[]"):
            if NAME_UNIT in properties:
                raise ValueError(
                    "There already is a unit identifier '%s' in index '%s'"
                    % (properties[NAME_UNIT], item)
                )

            properties[NAME_UNIT] = ureg.Quantity(tt)
        elif tt in ["-"]:
            # this is a list of stuff we'd like to ignore
            pass
        elif NAME_TYPE in properties:
            raise ValueError(
                "There already is a type identifier '%s' in index '%s'"
                % (properties[NAME_TYPE], item)
            )
        else:
            properties[NAME_TYPE] = tt

    for kk in [kk for kk in [NAME_UNIT, NAME_TYPE] if kk not in properties]:
        raise ValueError("Missing definition of '%s' in '%s'" % (kk, item))

    return properties


def hfss_interpret_dataframe(dataframe):
    """
    Process a dataframe as imported from `hfss_read_csv_to_dataframe`.

    The tabular data will be analyzed regarding the row and column header.
    HFSS stores the full set of parameters, as well as the physical unit and
    data type in those headers, what allows to exactly determine every aspect
    of every data value in the table.

    Since HFSS implicitly encodes SI unit information with magnitude scaling
    (milli/kilo/...) for every parameter and data row/column, it is imperative
    to handle units from the beginning to ensure proper readouts. Depending on
    the order of magnitude, these scalings are adaptive and subject to change
    throughout a csv table.

    Therefore, the HFSS importer keeps track of SI-unit-annotated magnitude
    information and acknowledges value scaling whenever necessary, especially
    when merging values to larger structural entities (up to the upmost level,
    which represents the tensor itself).

    Since it is possible to store full flattened tensors, including parameter
    annotations, using the HFSS format for .csv storage, it is very well
    possible to represent all information necessary to generate an EADF object
    in only one .csv file.

    As internal data representation, this function outputs a set of slices into
    the tensor found in the .csv file, called `subtable`s. In order to
    accomplish this task efficiently, all parameters present in row and column
    headers are analyzed in a first step. Every parameter will represent one
    dimension of the resulting tensor. Based on the variety of the parameters
    found, the two largest are selected to store the `subtable` representations
    with, thus minimizing the total number of such tables, which is given by
    the combinations of all values for all other parameters found.

    Each `subtable` contains one 2D unit-annotated ndarray. For both axes of
    that array the recorresponding support vectors (also including their SI
    units) are also stored within the subtable. Equally, the SI unit and the
    type of data (e.g. "real component of E field strength") are also stored
    for each subtable.

    A `subtable` is a :class:`dict` encoding this information as a set of
    distinct key names:

      - `data` The 2D array, representing one 2D slice of the tensor
        for one distinct parameter configuration of the remaining dimensions.
      - `unit` The SI unit of the elements in `data`.
      - `type` The data type of the elements in `data`.
      - `col_name` The name of the parameter representing axis 1 of `data`.
      - `col_indices` The support vector (with SI units), corresponding to
        `col_name`.
      - `row_name` The name of the parameter representing axis 0 of `data`.
      - `row_indices` The support vector (with SI units), corresponding to
        `row_name`.

    Parameters
    ----------
    dataframe : DataFrame
        The dataframe to be processed

    Returns
    -------
    list of dict
        The `subtable` representation of the data in `dataframe`. See the
        description above for more information regarding the data layout.
    """

    # extract the properties of the column axis from the column descriptor
    # type refers to the variable that is varied along this axis
    # unit refers to the unit corresponding to the columns
    column_properties = hfss_interpret_header(dataframe.columns.name)

    # extract the set of properties defined in each row from the row headers
    # and directly embed the data entries for that row with the corresponding
    # unit
    dataframe_rows = [hfss_interpret_header(ii) for ii in dataframe.index]
    arr_data = np.array(dataframe)
    for rr, row in enumerate(dataframe_rows):
        unit = row.pop(NAME_UNIT)
        row[NAME_DATA] = arr_data[rr, :] * unit

    # determine which properties are defined in the rows
    keys = set()
    for rr in dataframe_rows:
        keys.update(rr.keys())

    # remove the actual data field from analysis
    keys.remove(NAME_DATA)
    keys = list(keys)

    # now look at which of the keys have how many different values
    incidence = {key: set([]) for key in keys}
    for rr in dataframe_rows:
        for key, value in rr.items():
            if key != NAME_DATA:
                incidence[key].add(value)

    # Every key with just one incident value can be extracted to a common
    # parameter set since we already know the column axis information we may
    # add that directly to the common parameter dictionary such that we can
    # easily compile the full table information later-on.
    common = {
        NAME_COL_NAME: column_properties[NAME_TYPE],
        NAME_COL_INDICES: (
            np.array(dataframe.columns) * column_properties[NAME_UNIT]
        ),
    }
    for key, values in incidence.items():
        if len(values) == 1:
            common[key] = dataframe_rows[0][key]
            keys.remove(key)
            for rr in dataframe_rows:
                del rr[key]

    # Now remove these items from the incidence array
    incidence = {
        key: values for key, values in incidence.items() if len(values) > 1
    }

    if len(incidence) < 1:
        raise ValueError("Here's something wrong with the table data")

    # Now select the parameter with the most different values and use this as
    # our second axis also, add the row name information as axis name to the
    # common dictionary.
    counts = {key: len(values) for key, values in incidence.items()}
    row_name = [
        key for key, count in counts.items() if count == max(counts.values())
    ][0]
    common[NAME_ROW_NAME] = row_name
    del incidence[row_name]

    # Now we are left with incidents, which is a dictionary of key names and
    # all their possible values that they may represent. Therefore we'll need
    # a Cartesian product of this to determine the values we then need to
    # compile from the original table.
    # If we're done with that, add the common information to the table
    subtables = [
        dict(zip(incidence, values)) for values in product(*incidence.values())
    ]
    for table in subtables:
        table.update(common)

    # now all that there is left to say: "Bring in the nuns!"
    for table in subtables:
        subrows = [
            row
            for row in dataframe_rows
            if all([table[key] == row[key] for key in incidence.keys()])
        ]

        if len(subrows) < 1:
            continue

        # verify that all rows have the same length
        shape = subrows[0][NAME_DATA].shape
        if not all([row[NAME_DATA].shape == shape for row in subrows]):
            raise ValueError("Not all rows in one subtable have same length.")

        # The first element selects the unit we use for describing all tables.
        # If any of the other rows are incompatible to that, throw
        # an exception.
        unit_data = subrows[0][NAME_DATA].u
        unit_row = subrows[0][row_name].u
        table[NAME_DATA] = (
            np.vstack([row[NAME_DATA].to(unit_data).m for row in subrows])
            * unit_data
        )
        table[NAME_ROW_INDICES] = (
            np.hstack([row[row_name].to(unit_row).m for row in subrows])
            * unit_row
        )

    return subtables


def hfss_determine_subtable_types_and_parameters(*subtables):
    """
    Analyze a set of subtables for the parameters that are encoded therein.

    Parameters
    ----------
    *subtables : multipledict
        A variable number of :py:obj:`subtables` structures. For more
        information on :py:obj:`subtables`s, please consult
        :py:obj:`hfss_interpret_dataframe`.

    Returns
    -------
    (dict, dict)
        Two dictionaries will be returned.
        The first dictionary contains the set of all data types found within
        the set of all given `subtable`s. The key name of each entry denotes
        the type name and its value the corresponding SI unit, as encountered.
        The second dictionary contains the set of all parameters, accordingly.
        Here, the value of an entry corresponds to the superset of all distinct
        parameter values encountered within the given set of `subtable`s.
        Every of these support values must not necessarily be present in all
        of the subtables. However, every occurance of the parameter name among
        all `subtable` entries is represented in this support array.
        Proper handling of varying units or unit magnitudes across multiple
        `subtable`s is maintained for the resulting support vector.
    """
    reserved_keys = [
        NAME_TYPE,
        NAME_DATA,
        NAME_COL_NAME,
        NAME_COL_INDICES,
        NAME_ROW_NAME,
        NAME_ROW_INDICES,
    ]
    parameters = {}
    types = {}
    for subtable in subtables:
        if not all((key in subtable for key in reserved_keys)):
            raise ValueError("Not all required keys are present in subtable.")

        def append_parameter_range(name, indices):
            parameters[name] = (
                indices
                if name not in parameters
                else np.unique(
                    np.hstack(
                        [parameters[name].m, indices.to(parameters[name].u).m]
                    )
                )
                * parameters[name].u
            )

        if subtable[NAME_TYPE] not in types:
            types[subtable[NAME_TYPE]] = subtable[NAME_DATA].u

        append_parameter_range(
            subtable[NAME_COL_NAME], subtable[NAME_COL_INDICES]
        )
        append_parameter_range(
            subtable[NAME_ROW_NAME], subtable[NAME_ROW_INDICES]
        )

        for key in subtable:
            if key not in reserved_keys:
                append_parameter_range(key, subtable[key])

    # now sort the indices for each parameter
    for indices in parameters.values():
        indices.sort()

    return types, parameters


# ##################################### Actual importer
def fromHFSS(*filenames, **options):
    """
    Importer from HFSS .csv exports.

    This format allows to specify a list of angles (azi, ele)_i and
    beam pattern values v_i = (pol, freq, elem)_i which are then
    interpolated along the two angular domains to get a regular grid in
    azimuth and co-elevation. By default this is done using vector spherical
    harmonics, since they can deal with irregular sampling patterns
    quite nicely. In this format for each angular sampling point, we
    need to have excited the array elements with the same frequencies.

    Parameters
    ----------
    *filenames : str
        One or multiple filenames, corresponding to .csv HFSS output files.
    key_CoElevation : str, optional
        The HFSS parameter name corresponding to the Azimuth dimension.

        Defaults to `Theta`.
    key_Azimuth : str, optional
        The HFSS parameter name corresponding to the Co-Elevation dimension.

        Defaults to `phi`.
    key_Frequency : str, optional
        The HFSS parameter name corresponding to the frequency dimension.

        Defaults to `freq`.
    key_Polarization : tuple of str, optional
        An immutable tuple of HFSS type identifiers, that shall populate the
        polarization dimension of the EADF tensor.

        Defaults to `('rEPhi', 'rETheta')`.
    position : :class:`numpy.ndarray`, optional
        The position of the antenna, given in [3 x 1] coordinates.
    **options : kwargs
        Further key-worded arguments will be passed on to the construction of
        the EADF object

    Returns
    -------
    EADF
        Created Array
    """
    # pop the importer options
    key_CoElevation = options.pop("key_CoElevation", "Theta")
    key_Azimuth = options.pop("key_Azimuth", "phi")
    key_Frequency = options.pop("key_Frequency", "freq")
    key_Polarization = options.pop("key_Polarization", ("rEPhi", "rETheta"))
    arr_Position = options.pop("position", np.zeros((3, 1)))

    subtables = []
    for filename in filenames:
        subtables.extend(
            hfss_interpret_dataframe(hfss_read_csv_to_dataframe(filename))
        )

    # extract which data types and parameters (including the occuring index
    # values for that parameters) occur in the data
    typedefs, parameters = hfss_determine_subtable_types_and_parameters(
        *subtables
    )

    # typedefs:
    #   - Available data types (which may stack to the "Polarization" dimension
    #     of the tensor)
    #   - May be specified using multiple definitions
    #   - Dictionary that maps the corresponding unit type to the type name
    #
    # parameters:
    #   - Available dimensions (which have one or more index values) in the
    #     tensor data
    #   - Dictionary that maps all the indices that were used for this
    #     parameter to the parameter name

    # now check that we have enough parameters to populate the tensor.
    # Throw errors if we have more, since we do not know which of those
    # realizations we shall use. Later on we could support to select from them
    # using additional keyworded arguments. Later on...
    required_keys = [
        key
        for key in (key_CoElevation, key_Azimuth, key_Frequency)
        if key is not None
    ]
    keys_not_found = [key for key in required_keys if key not in parameters]
    keys_unused = [key for key in parameters if key not in required_keys]
    if len(keys_not_found) > 0:
        raise ValueError(
            (
                "Required keys '%s' not found in available parameters of "
                + "imported data %s."
            )
            % (keys_not_found, str(tuple(parameters.keys())))
        )

    if len(keys_unused) > 0:
        raise ValueError(
            (
                "Excess parameters '%s' in available set of parameters of "
                + "imported data %s"
            )
            % (keys_unused, str(tuple(parameters.keys())))
        )

    # then, check if we have all the data types present, which shall then be
    # used to populate the tensor.
    # 1. We collect all definitions for each type.
    # 2. Do throw an error here if a type is not found.
    #    Do not throw an error here, if more than the requested set of types is
    #    found, since we can clearly omit data that we do not use at all.
    #    However, we must support the syntax to write out complex types as
    #    're'/'im' or 'mag'/'phase'.
    # 3. We check that we either have
    #     - a single definition of that type name  -- OR --
    #     - a 're'/'im' definition  -- OR --
    #     - a 'mag'/'phase' definition
    #    If we find mixtures of those definitions, we throw an error.
    # 4. If we find only one of a paired definition (e.g. only 're' without
    #    'im'), we throw a warning.
    types = {}

    # (1.)
    for typedef in typedefs.keys():
        parsed = parse_parantheses(typedef)

        def add_typedef(typename, typedef):
            container = types.setdefault(typename, set([]))
            if typedef in container:
                raise ValueError(
                    "Multiple Definitions for '%s' in '%s'"
                    % (typedef, typename)
                )
            else:
                container.add(typedef)

        if len(parsed) == 2 and len(parsed[1]) == 1:
            add_typedef(parsed[1][0], parsed[0])
        elif len(parsed) == 1:
            add_typedef(parsed[0], None)
        else:
            raise ValueError(
                "Unsupported data type definition '%s'" % (typedef,)
            )

    # (2.)
    required_types = [] if key_Polarization is None else key_Polarization
    types_not_found = [key for key in required_types if key not in types]
    if len(types_not_found) > 0:
        raise ValueError(
            (
                "Required data types '%s' not defined in available types of "
                + "imported data %s."
            )
            % (types_not_found, str(tuples(types.keys())))
        )

    # limit all further work to only the required data type definitions
    types = {
        key: value for key, value in types.items() if key in required_types
    }

    # (3.)
    for typename, descriptors in types.items():
        fully_defined = any(
            [descriptors == descriptor for descriptor in DESC_ALL]
        )
        partially_defined = any(
            [descriptors <= set(descriptor) for descriptor in DESC_ALL]
        )
        if not partially_defined:
            raise ValueError(
                (
                    "Definition of data type '%s' using '%s' "
                    + "not supported (yet)."
                )
                % (typename, descriptors)
            )
        elif not fully_defined:
            warnings.warn(
                "Data type '%s' only defined partially ('%s')"
                % (typename, descriptors)
            )

    # Now the structure of the available data has been checked to the point
    # where we know whether a tensor may be constructed fully. (We still
    # neither do know if it is fully defined, nor if the units match)

    # Let's construct subtensors for each data type definition found ...
    shape = tuple(
        len(parameters[key])
        for key in [key_CoElevation, key_Azimuth, key_Frequency]
    )
    subtensors = {typedef: np.full(shape, np.NaN) for typedef in typedefs}

    # ... define how to handle the support along the dimensions ...
    support_CoElevation = parameters[key_CoElevation]
    support_Azimuth = parameters[key_Azimuth]
    support_Frequency = parameters[key_Frequency]

    def determine_indices_and_axis(support, subtable, keyname):
        if subtable[NAME_COL_NAME] == keyname:
            support_values = np.atleast_1d(
                subtable[NAME_COL_INDICES].to(support.u).m
            )
            axis = 1
        elif subtable[NAME_ROW_NAME] == keyname:
            support_values = np.atleast_1d(
                subtable[NAME_ROW_INDICES].to(support.u).m
            )
            axis = 0
        else:
            support_values = np.atleast_1d(subtable[keyname].to(support.u).m)
            axis = 2

        indices = np.atleast_1d(np.searchsorted(support.m, support_values))
        if not np.array_equal(support_values, support.m[indices]):
            raise ValueError(
                (
                    "Could not determine correct indices for "
                    + "values '%s' in '%s'"
                )
                % (support_values, support.m)
            )

        return indices, axis

    # ... and populate them
    for typename in subtensors.keys():
        # First, collect all subtables that describe this subtensor
        subtensor_tables = [
            subtable
            for subtable in subtables
            if subtable[NAME_TYPE] == typename
        ]
        if len(subtensor_tables) < 1:
            raise RuntimeError(
                (
                    "Tensor descriptor for '%s' is empty. "
                    + "This is probably a bug."
                )
                % (typename,)
            )

        # Then, from the first subtensor slice, assign the unit for this
        # subtensor. This unit will be used during further tensor population
        assigned = np.zeros(shape, dtype="bool")
        subtensors[typename] = (
            subtensors[typename] * subtensor_tables[0][NAME_DATA].u
        )

        # Now iterate to fill the subtensor with the corresponding slices and
        # correct for unit mismatches
        subtensor = subtensors[typename]
        for subtable in subtensor_tables:
            # Determine the indices into the tensor, as given by the fixed
            # support in the tensor dimensions
            idx_CoElevation, axis_CoElevation = determine_indices_and_axis(
                support_CoElevation, subtable, key_CoElevation
            )
            idx_Azimuth, axis_Azimuth = determine_indices_and_axis(
                support_Azimuth, subtable, key_Azimuth
            )
            idx_Frequency, axis_Frequency = determine_indices_and_axis(
                support_Frequency, subtable, key_Frequency
            )

            # Now permute the axes of subtensor such, that we may address them
            # in the way that matches to the data array in subtable
            indices = (idx_CoElevation, idx_Azimuth, idx_Frequency)
            axes = (axis_CoElevation, axis_Azimuth, axis_Frequency)
            remapping = np.argsort(np.array(axes))
            if set(remapping) != set(np.arange(len(remapping))):
                raise ValueError(
                    "Mapping of subtable support into the support of the "
                    + "tensor failed."
                )
            indices_section = tuple(np.array(indices)[remapping])
            if all(
                [
                    indices_section[ii].size != 1
                    for ii in range(2, len(indices_section))
                ]
            ):
                raise ValueError(
                    "A subtable contains more than two active data dimensions."
                )

            # We can now safely assume that dimensions two and above only do
            # not contain index ranges. Therefore we may omit them from the
            # view we are about to generate.

            subtensor_section = np.transpose(subtensor.m, axes)
            assigned_section = np.transpose(assigned, axes)
            for ii in reversed(range(2, len(indices_section))):
                subtensor_section = subtensor_section[
                    ..., indices_section[ii][0]
                ]
                assigned_section = assigned_section[
                    ..., indices_section[ii][0]
                ]

            if np.any(assigned_section):
                ValueError("Some indices in subtensor were assigned twice.")

            # Convert the slice data to the unit the subtensor is in and apply
            # the data column-by-column.
            data = subtable[NAME_DATA].to(subtensor.u).m
            for ii, cc in enumerate(indices_section[1]):
                subtensor_section[indices_section[0], cc] = data[:, cc]
                assigned_section[indices_section[0], cc] = True

            if not np.all(assigned_section):
                raise ValueError("Writing subtable data into tensor failed.")

        # Now check that this subtensor has been populated fully.
        if not np.all(assigned):
            warn(
                "Tensor data for descriptor '%s' was not complete"
                % (typename,)
            )

        if np.any(np.isnan(subtensor)):
            warn(
                (
                    "Tensor data for descriptor '%s' contains undefined "
                    + "elements (NaN)"
                )
                % (typename,)
            )

    # Now, since we have the subtensors, we want to compose them to the actual
    # thing. [Co-Elevation, Azimuth, Frequency, Polarization
    def fetch_subtensor_description(key, descriptor, unit):
        return (
            subtensors["%s(%s)" % (descriptor, key)].to(unit)
            if descriptor in types[key]
            else None
        )

    tensor = (
        np.full(shape + (len(key_Polarization),), np.NaN, dtype=np.float64)
        * list(subtensors.values())[0].u
    )
    for pp, key in enumerate(key_Polarization):
        if types[key] == DESC_REAL:
            tensor[:, :, :, pp] = subtensors[key].to(tensor.u)
        elif len(types[key] & DESC_COMPLEX) > 0:
            if not np.iscomplex(tensor.dtype):
                tensor = tensor.m.astype(np.complex) * tensor.u

            ten_real = fetch_subtensor_description(key, NAME_REAL, tensor.u)
            ten_imaginary = fetch_subtensor_description(
                key, NAME_IMAGINARY, tensor.u
            )
            if ten_real is not None:
                tensor[:, :, :, pp] = (
                    ten_real
                    if ten_imaginary is None
                    else ten_real + 1j * ten_imaginary
                )
            elif ten_real is not None:
                tensor[:, :, :, pp] = ten_imaginary
            else:
                raise ValueError(
                    "Descriptor handling out of sync for polarization '%s'"
                    % (key,)
                )

        elif len(types[key] & DESC_POLAR) > 0:
            if not np.iscomplex(tensor.dtype):
                tensor = tensor.m.astype(np.complex) * tensor.u

            ten_magnitude = fetch_subtensor_description(
                key, NAME_MAGNITUDE, tensor.u
            )
            ten_phase = fetch_subtensor_description(key, NAME_PHASE, tensor.u)
            tensor[:, :, :, pp] = (
                ten_magnitude if ten_magnitude is not None else (1.0 + 0.0j)
            )
            if ten_phase is not None:
                tensor[:, :, :, pp] *= exp(1j * ten_phase)
        else:
            raise ValueError(
                "Unknown subtensor descriptor '%s'" % (types[key],)
            )

    # Finally, check if the tensor is properly populated (and contains no NaNs)
    if np.any(np.isnan(tensor)):
        warn("Tensor data not complete, contains NaNs")

    # Since the tensor data is collected now, we make sure that the support
    # matches the EADF requirements. If there are non-radians along the Azimuth
    # and CoElevation dimensions, they need to be converted now.
    # Then, finally, let's construct the EADF object
    return EADF(
        tensor[..., np.newaxis],
        support_CoElevation.to("radian"),
        support_Azimuth.to("radian"),
        support_Frequency,
        arr_Position,
        **options
    )
