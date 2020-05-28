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
This class is the central object where everything else is centered around. It
has convenient methods and properties to handle:

 - Fourier interpolation along Co-Elevation and Azimuth of the provided
   measurement data, see :py:obj:`eadf.eadf.EADF.pattern`,
 - Fourier interpolation of the measurement data's first and second order
   derivatives, see :py:obj:`eadf.eadf.EADF.gradient` and
   :py:obj:`eadf.eadf.EADF.hessian`,
 - Spline interpolation along excitation frequency of the provided measurement
   data, see :py:obj:`eadf.eadf.EADF.operatingFrequencies`,
 - reduction of the coefficients used for the Fourier interpolation to allow
   denoising and/or faster computations, see
   :py:obj:`eadf.eadf.EADF.blockSize` and
   :py:obj:`eadf.eadf.EADF.optimizeBlockSize`,
 - convenient definition of stationary subbands via
   :py:obj:`eadf.eadf.EADF.defineSubBands` and
   :py:obj:`eadf.eadf.EADF.subBandIntervals`,
 - execution control of the calculations with respect to used datatypes, lower
   memory consumption and the used computation device, see
   :py:obj:`eadf.backend` and
 - easy storage and loading from disk, see :py:obj:`eadf.eadf.EADF.save` and
   :py:obj:`eadf.eadf.EADF.load`.
"""

import numpy as np
import logging
import pickle

from . import __version__
from .core import evaluatePattern
from .core import evaluateGradient
from .core import evaluateHessian
from .core import calcBlockSize
from .core import ureg
from .preprocess import splineInterpolateFrequency
from .preprocess import sampledToFourier
from .preprocess import symmetrizeData
from .preprocess import setCompressionFactor
from .backend import lowMemory
from .backend import dtype
from .backend import rDtype
from .backend import cDtype

__all__ = ["EADF"]


def _checkInputData(
    arrData: np.ndarray,
    arrCoEle: np.ndarray,
    arrAzi: np.ndarray,
    arrFreq: np.ndarray,
    arrPos: np.ndarray,
    **options
) -> None:
    def check_support(
        arrSupport,
        strName,
        targetUnit,
        expectedShape,
        cast_shape=np.atleast_1d,
    ):

        # Cast support array to default unit, if not already a quantity
        if not isinstance(arrSupport, ureg.Quantity):
            arrSupport = arrSupport * ureg.Quantity(targetUnit).u

        # Make any ndarray, with only one axis of size > 1, a vector
        arrSupport = cast_shape(
            np.squeeze(arrSupport.to(targetUnit).m)
        ) * ureg.Quantity(targetUnit)

        # check if sizes match
        if arrSupport.shape != expectedShape:
            raise ValueError(
                "EADF:%s.shape%s != expected shape%s"
                % (strName, repr(arrSupport.shape), repr(expectedShape))
            )

        return arrSupport

    # check for correct shapes and support and introduce default units
    arrCoEle = check_support(
        arrCoEle, "arrCoEle", "radian", (arrData.shape[0],)
    )
    arrAzi = check_support(arrAzi, "arrAzi", "radian", (arrData.shape[1],))
    arrFreq = check_support(arrFreq, "arrFreq", "hertz", (arrData.shape[2],))
    arrPos = check_support(
        arrPos,
        "arrPos",
        "meter",
        (3, arrData.shape[4]),
        cast_shape=lambda arr: (
            arr.reshape((-1, 1)) if arr.ndim == 1 else arr
        ),
    )

    if not isinstance(arrData, ureg.Quantity):
        arrData = arrData * ureg.Quantity("1").u

    # Now we want to allow for some preparations or modifications of the
    # data given with the input arguments, prior to asserting correctness
    # or alignment of the grids.

    def preprocess_support_axis(arrSupport, nameOption):
        func = options.get(nameOption, None)
        return arrSupport if func is None else func(arrSupport)

    def rearrange_axis(arrSupport, arrData, numAxis, arrIndex):
        index = arrData.ndim * [slice(None)]
        index[numAxis] = arrIndex
        return arrSupport[arrIndex], arrData[tuple(index)]

    # Preprocessing: Apply lambda function to axis support
    arrCoEle = preprocess_support_axis(arrCoEle, "modify_support_CoEle")
    arrAzi = preprocess_support_axis(arrAzi, "modify_support_Azi")
    arrFreq = preprocess_support_axis(arrFreq, "modify_support_Freq")

    # Preprocessing: sort tensor align the support axes
    if options.get("sort_support", False):
        arrCoEle, arrData = rearrange_axis(
            arrCoEle, arrData, 0, np.argsort(arrCoEle)
        )
        arrAzi, arrData = rearrange_axis(
            arrAzi, arrData, 1, np.argsort(arrAzi)
        )
        arrFreq, arrData = rearrange_axis(
            arrFreq, arrData, 2, np.argsort(arrFreq)
        )

    # Preprocessing: truncate along data axis based on support
    def truncate_axis(arrSupport, arrData, numAxis, nameOption):
        limits = options.get(nameOption, None)
        if limits is None:
            return arrSupport, arrData

        if len(limits) != 2 or limits[0] > limits[1]:
            raise ValueError("EADF:Truncation range must be an interval")

        return rearrange_axis(
            arrSupport,
            arrData,
            numAxis,
            np.logical_and(arrSupport >= limits[0], arrSupport <= limits[1]),
        )

    arrCoEle, arrData = truncate_axis(arrCoEle, arrData, 0, "truncate_CoEle")
    arrAzi, arrData = truncate_axis(arrAzi, arrData, 1, "truncate_Azi")
    arrFreq, arrData = truncate_axis(arrFreq, arrData, 2, "truncate_Freq")

    # check if angular are sampled regularly
    # and the are also sorted in ascending order
    def check_grid(**support):
        for name, arrSupport in support.items():
            arrDiff = np.diff(arrSupport.m)

            if not np.isclose(arrDiff.max(), arrDiff.min()):
                raise ValueError(
                    "EADF:%s grid is not sampled evenly" % (name,)
                )

            if np.any(arrDiff <= 0):
                raise ValueError("EADF:%s grid must be sorted." % (name,))

    check_grid(azimuth=arrAzi, coelevation=arrCoEle)

    # in elevation we check if we sampled from north to south pole
    if not np.allclose(arrCoEle.m[0], 0):
        raise ValueError("EADF:you must sample at the north pole.")

    if not np.allclose(arrCoEle.m[-1], np.pi):
        raise ValueError("EADF:you must sample at the south pole.")

    return (arrData, arrCoEle, arrAzi, arrFreq, arrPos)


class EADF(object):
    def _eval(
        self, arrCoEle: np.ndarray, arrAzi: np.ndarray, funCall,
    ) -> np.ndarray:
        """Unified Evaluation Function

        This function allows to calculate the Hessian, Jacobian
        and the values themselves with respect to the parameters
        angle and frequency.

        Parameters
        ----------
        arrCoEle : np.ndarray
            Sample at these elevations in radians
        arrAzi : np.ndarray
            Sample at these azimuths in radians
        funCall : function
            evaluatePattern, evaluateGradient, evaluateHessian

        Returns
        -------
        np.ndarray
            pattern, gradient or Hessian array
        """

        if arrCoEle.shape[0] != arrAzi.shape[0]:
            raise ValueError(
                "eval: supplied angle arrays have size %d and %d."
                % (arrCoEle.shape[0], arrAzi.shape[0])
            )

        # convert the given inputs to the current datatype
        # nothing is copied, if everything is already on the host / the
        # GPU respectively
        arrCoEle = np.asarray(arrCoEle.astype(rDtype))
        arrAzi = np.asarray(arrAzi.astype(rDtype))

        # print(self.arrDataCalc.shape)
        # print(self.muCoEleCalc.shape)
        # print(self.muAziCalc.shape)

        # calls either pattern, gradient or hessian
        res = funCall(
            arrCoEle,
            arrAzi,
            self.muCoEleCalc,
            self.muAziCalc,
            self.arrDataCalc,
            self._blockSize,
            self._lowMemory,
        )

        if self.subBandIntervals is None:
            return res
        else:
            return res[:, self.subBandAssignment]

    def pattern(self, arrCoEle: np.ndarray, arrAzi: np.ndarray) -> np.ndarray:
        """Sample the Beampattern at given Angles

        The supplied arrays need to have the same length. The returned
        array has again the same length. This method samples the EADF object
        for given angles at operating frequencies, all polarizations and array
        elements. So it yields a Ang x Freq x Pol x Element ndarray.

        .. note::
          If the GPU is used for calculation a cupy.ndarray is returned,
          so for further processing on the host, you need to copy ot yourself.
          otherwise you can simply continue on the GPU device. Moreover,
          if you supply cupy.ndarrays with the right data types,
          this also speeds up the computation, since no copying or
          conversion have to be done.

        Parameters
        ----------
        arrCoEle : np.ndarray
            Sample at these elevations in radians
        arrAzi : np.ndarray
            Sample at these azimuths in radians

        Returns
        -------
        np.ndarray
            (Ang x Freq x Pol x Elem) result array
        """
        return self._eval(arrCoEle, arrAzi, evaluatePattern)

    def gradient(self, arrCoEle: np.ndarray, arrAzi: np.ndarray) -> np.ndarray:
        """Sample the Beampattern Gradient at given Angles

        The supplied arrays need to have the same length. The returned
        array has again the same length. This method samples the EADF object
        for given angles at operating frequencies, all polarizations and array
        elements. So it yields a Ang x Freq x Pol x Element ndarray.

        .. note::
          If the GPU is used for calculation a cupy.ndarray is returned,
          so for further processing on the host, you need to copy ot yourself.
          otherwise you can simply continue on the GPU device. Moreover,
          if you supply cupy.ndarrays with the right data types,
          this also speeds up the computation, since no copying or
          conversion have to be done.

        Parameters
        ----------
        arrCoEle : np.ndarray
            Sample at these elevations in radians
        arrAzi : np.ndarray
            Sample at these azimuths in radians

        Returns
        -------
        np.ndarray
            (Ang x Freq x Pol x Elem x 2)
        """
        return self._eval(arrCoEle, arrAzi, evaluateGradient)

    def hessian(self, arrCoEle: np.ndarray, arrAzi: np.ndarray) -> np.ndarray:
        """Sample the Beampattern Hessian at given Angles

        The supplied arrays need to have the same length. The returned
        array has again the same length. This method samples the EADF object
        for given angles at operating frequencies, all polarizations and array
        elements. So it yields a Ang x Freq x Pol x Element ndarray.

        .. note::
          If the GPU is used for calculation a cupy.ndarray is returned,
          so for further processing on the host, you need to copy ot yourself.
          otherwise you can simply continue on the GPU device. Moreover,
          if you supply cupy.ndarrays with the right data types,
          this also speeds up the computation, since no copying or
          conversion have to be done.

        Parameters
        ----------
        arrCoEle : np.ndarray
            Sample at these elevations in radians
        arrAzi : np.ndarray
            Sample at these azimuths in radians

        Returns
        -------
        np.ndarray
            (Ang x Freq x Pol x Elem x 2 x 2) result array,
            hermitian along the last two 2x2 dimensions.
        """
        return self._eval(arrCoEle, arrAzi, evaluateHessian)

    @property
    def arrIndAziCompress(self) -> np.ndarray:
        """Subselection indices for the compressed array in azimuth

        This is influenced by :py:obj:`eadf.eadf.EADF.compressionFactor`

        .. note::

          This property is cacheable.

        Returns
        -------
        np.ndarray
            Subselection in spatial Fourier domain in azimuth

        """
        if self._arrIndAziCompress is None:
            (
                self._arrIndCoEleCompress,
                self._arrIndAziCompress,
            ) = setCompressionFactor(
                self.arrFourierData,
                self._numCoEleInit,
                self._numAziInit,
                self._compressionFactor,
            )
        return self._arrIndAziCompress

    @property
    def arrIndCoEleCompress(self) -> np.ndarray:
        """Subselection indices for the compressed array in elevation (ro)

        This is influenced by :py:obj:`eadf.eadf.EADF.compressionFactor`.

        .. note::

          This property is cacheable.

        Returns
        -------
        np.ndarray
            Subselection in spatial Fourier domain in elevation

        """
        if self._arrIndCoEleCompress is None:
            (
                self._arrIndCoEleCompress,
                self._arrIndAziCompress,
            ) = setCompressionFactor(
                self.arrFourierData,
                self._numCoEleInit,
                self._numAziInit,
                self._compressionFactor,
            )
        return self._arrIndCoEleCompress

    def truncateCoefficients(self, numCoEle: int, numAzi: int) -> None:
        """Manually Truncate the Fourier Coefficients

        Parameters
        ----------
        numCoEle : int
            Truncation size in co-elevation
        numAzi : int
            Truncation size in azimuth
        """
        if numCoEle > self._numCoEleInit:
            raise ValueError("numCoEle cannot be larger than data")
        if numAzi > (self._numAziInit // 2):
            raise ValueError("numAzi cannot be larger than data")
        if numCoEle < 0:
            raise ValueError("numCoEle cannot be smaller than 0")
        if numAzi < 0:
            raise ValueError("numAzi cannot be smaller than 0")

        self._arrIndCoEleCompress = np.fft.fftshift(
            np.arange(self._numCoEleInit)
        )
        if numCoEle > 0:
            self._arrIndCoEleCompress = self._arrIndCoEleCompress[
                numCoEle:-numCoEle
            ]
        self._arrIndAziCompress = np.fft.fftshift(np.arange(self._numAziInit))
        if numAzi > 0:
            self._arrIndAziCompress = self._arrIndAziCompress[numAzi:-numAzi]

        self._arrDataCalc = None
        self._muCoEleCalc = None
        self._muAziCalc = None

    @property
    def arrAzi(self) -> np.ndarray:
        """Return Array Containing the sampled Azimuth Angles

        .. note::

          This property is read only.

        Returns
        -------
        np.ndarray
            Sampled Azimuth Angles in radians

        """
        return self._arrAzi

    @property
    def arrCoEle(self) -> np.ndarray:
        """Return Array Containing the sampled Co-Elevation Angles

        .. note::

          This property is read only.

        Returns
        -------
        np.ndarray
            Sampled Co-Elevation Angles in radians

        """
        return self._arrCoEle

    @property
    def arrFreq(self) -> np.ndarray:
        """Return Array Containing the Sampled Frequencies

        .. note::

          This property is read only.

        Returns
        -------
        np.ndarray
            Sampled Frequencies in Hertz

        """
        return self._arrFreq

    @property
    def numElements(self) -> int:
        """Number of Array Elements

        .. note::

          This property is read only.

        Returns
        -------
        int
            Number of Antenna Elements / Ports

        """
        return self._numElements

    @property
    def arrPos(self) -> np.ndarray:
        """Positions of the Elements as 3 x numElements

        Cartesian coordinates.

        .. note::

          This property is read only.

        Returns
        -------
        np.ndarray
            Positions of the Elements as 3 x numElements

        """

        return self._arrPos

    @property
    def arrFourierData(self) -> np.ndarray:
        """Return the Fourier Data used to represent the antenna.

        .. note::

          It is important to know that this property's meaning depends on the
          fact if the user has defined stat. subbands or not. If there are no
          defined subbands, this property returns the Fourier data that is
          used to calculate the beampatterns at the specified
          :py:obj:`eadf.eadf.EADF.operatingFrequencies`. In this case it is
          the result of an interpolation process along the measured
          frequencies.

          In case when there are stationary subbands, each slice along axis=2
          of this property represents the data that is used to represent the
          beampattern on the subband with the same slice index.

        .. note::

          This property is read only.

        Returns
        -------
        np.ndarray
            CoEle x Azi x Freq x Pol x Port shaped Fourier Data
        """
        if self._arrFourierData is None:
            if self._subBandIntervals is None:
                # if we have only one frequency, we do no interpolation,
                # or anything.
                if self.arrFreq.shape[0] == 1:
                    self._arrFourierData = self._arrInputFourierData
                else:
                    self._arrFourierData = self._interpolateFourierData()
            else:
                self._arrFourierData = self._subBandData

        return self._arrFourierData

    def _interpolateFourierData(self) -> np.ndarray:
        """Interpolate the Fourier Data along Frequency

        Given the currently set operating frequencies, we evaluate splines
        of order 5, which were generated from the measurement data.

        Returns
        -------
        np.ndarray
            CoEle x Azi x operatingFreq x Pol x Elem

        """
        res = np.empty(
            (
                *self._arrInputFourierData.shape[:2],
                self._operatingFrequencies.shape[0],
                *self._arrInputFourierData.shape[3:],
            ),
            dtype=cDtype,
        )

        # evaluate the spline
        for ii3 in range(res.shape[3]):
            for ii4 in range(res.shape[4]):
                for ii1 in range(res.shape[1]):
                    for ii0 in range(res.shape[0]):
                        # this might trigger spline generation, since
                        # self.frequencySplines is a cacheable property.
                        res[ii0, ii1, :, ii3, ii4] = np.array(
                            self.frequencySplines[ii0, ii1, 0, ii3, ii4](
                                self._operatingFrequencies
                            )
                        ) + 1j * np.array(
                            self.frequencySplines[ii0, ii1, 1, ii3, ii4](
                                self._operatingFrequencies
                            )
                        )
        return res

    @property
    def arrRawData(self) -> np.ndarray:
        """Return the Raw Data used during construction.

        This is the already correctly along Co-Elevation and Azimuth
        periodified data. It is basically the measurement data in such a
        form that we can Fourier transform it conveniently along Co-Elevation
        and azimuth.

        .. note::

          This property is read only.

        Returns
        -------
        np.ndarray
            Raw Data in 2 * Co-Ele x Azi x Freq x Pol x Element
        """

        return self._arrRawData

    @property
    def lowMemory(self) -> bool:
        """Does this EADF object operate in Low-Memory mode?

        Low memory mode can be switched on in order to split up the Antenna
        repsonse calculation in blocks along the requested angles. This way
        only the currently needed blocks in equation (7) in the EADF paper
        by Landman are used to calculate a block. It should always be used
        together with :py:obj:`eadf.eadf.EADF.blockSize` in order to maximize
        the possibly lower computation speed. It is set by the shell
        environment variable EADF_LOWMEM=1/0.

        .. note::

          This property is read cacheable.

        Returns
        -------
        bool
            Flag if low memory mode is switched on.

        """
        return self._lowMemory

    @property
    def dtype(self) -> str:
        """Data Type to use during calculations

        Especially if calculations are done on the GPU, switching to single
        floating precision can speed up calculations tremendously. Obviously
        one also saves some memory. It is set by the shell
        environment variable EADF_LOWMEM=single/double.

        .. note::

          This property is read cacheable.

        Returns
        -------
        str
            either 'single' for single precision or 'double' for
            double precision
        """

        return self._dtype

    @property
    def blockSize(self) -> int:
        """Block Size for the Evaluation Functions

        See :py:obj:`eadf.eadf.EADF.optimizeBlockSize` and
        :py:obj:`eadf.eadf.EADF.lowMem`.

        Returns
        -------
        int
            block size

        """
        return self._blockSize

    @blockSize.setter
    def blockSize(self, blocksize: int) -> None:
        self._blockSize = blocksize

    def optimizeBlockSize(self, maxSize: int) -> None:
        """Optimize the Blocksize during the Calculation

        Instead of processing all angles and (possibly) frequencies all at
        once, we process them in blocks. This can produce a decent speedup
        and makes the transform scale nicer with increasing number of angles.

        Simply call this function, which might take some time to determine the
        best block size. See :py:obj:`eadf.eadf.EADF.blockSize` and
        :py:obj:`eadf.eadf.EADF.lowMem`.

        Parameters
        ----------
        maxSize : int
            Largest Block Size?
        """

        if maxSize < 1:
            raise ValueError("maxSize mus be larger than 0")

        self.blockSize = calcBlockSize(
            self.muCoEleCalc,
            self.muAziCalc,
            self.arrDataCalc,
            maxSize,
            self._lowMemory,
        )

    @property
    def compressionFactor(self) -> float:
        """Compression Factor

        Returns
        -------
        float
            Compression factor in (0,1]

        """
        return self._compressionFactor

    @compressionFactor.setter
    def compressionFactor(self, numValue: float) -> None:
        """Set the Compression Factor

        The EADF allows to reduce the number of parameters of a given
        beampattern by reducing the number of Fourier coefficients.
        This should be done carefully, since one should not throw away
        necessary information. So, we define a compression factor 'p', which
        determines how much 'energy' the remaining Fourier coefficients
        contain.
        So we have the equation: E_c = p * E, where 'E' is the energy of the
        uncompressed array. See also :py:obj:`setCompressionFactor`.

        Parameters
        ----------
        numValue : float
            Factor to be set. Must be in (0,1]. The actual subselection
            is done such that the remaining energy is always greater or
            equal than the specified value, which minimizes the expected
            computation time.

        """
        if (numValue <= 0.0) or (numValue > 1.0):
            raise ValueError("Supplied Value must be in (0, 1]")
        else:
            (
                self._arrIndCoEleCompress,
                self._arrIndAziCompress,
            ) = setCompressionFactor(
                self.arrFourierData,
                self._numCoEleInit,
                self._numAziInit,
                numValue,
            )
            self._compressionFactor = numValue
            self._arrDataCalc = None
            self._muCoEleCalc = None
            self._muAziCalc = None

    @property
    def arrDataCalc(self) -> np.ndarray:
        """Calculation Data used for the pattern and derivatives

        This data is ready to be fed into the pattern and derivatives routine.
        As such it is the periodified measurement data Fourier transformed
        along Co-Elevation and azimuth, possibly interpolated along
        measurement frequency and restricted to the indices determined by the
        compression factor. Moreover it is already in the correct datatype and
        also has been transferred to the computation device if necessary.

        .. note::

          This property is cacheable.

        Returns
        -------
        np.ndarray
            Description of returned object.

        """
        if self._arrDataCalc is None:
            self._arrDataCalc = np.asarray(
                self.arrFourierData[:, self.arrIndAziCompress][
                    self.arrIndCoEleCompress
                ].astype(cDtype)
            )
        return self._arrDataCalc

    @property
    def muCoEleCalc(self) -> np.ndarray:
        """Spatial Frequencies along Co-Elevation

        These are already in the right data type, on the right device and
        subselected accoring to the chosen compression factor.

        .. note::

          This property is cacheable.

        Returns
        -------
        np.ndarray
            Description of returned object.

        """
        if self._muCoEleCalc is None:
            self._muCoEleCalc = np.asarray(
                self._muCoEle[self.arrIndCoEleCompress].astype(rDtype)
            )
        return self._muCoEleCalc

    @property
    def muAziCalc(self) -> np.ndarray:
        """Spatial Frequencies along Azimuth

        These are already in the right data type, on the right device and
        subselected accoring to the chosen compression factor.

        .. note::

          This property is cacheable.

        Returns
        -------
        np.ndarray
            Description of returned object.

        """
        if self._muAziCalc is None:
            self._muAziCalc = np.asarray(
                self._muAzi[self.arrIndAziCompress].astype(rDtype)
            )
        return self._muAziCalc

    def defineSubBands(self, intervals: np.ndarray, data: np.ndarray) -> None:
        """Define Stationary SubBands

        Stationary subband define intervals in frequency over which we can
        describe the antenna response reasonably well with one single set of
        CoEle x Azi x Pol x Port data values (not necessarily measurement
        data). This happens if the pattern changes only negligibly for the
        application at hand and as such can be considered constant/stationary.

        If now several operating frequencies reside in the same stationary
        subband, one can save memory and computation time when calculating the
        pattern or its derivatives.

        .. note::

          Using this function triggers the spline generation and the
          regeneration of all the data used for calculating the pattern and so
          forth. Be patient.

        Parameters
        ----------
        intervals : np.ndarray
            Array containing the intervals bounds. Its first element must
            coincide with arrFreq[0] and its last with arrFreq[-1].
            It must be sorted.
        data : np.ndarray
            Description of parameter `data`.

        """
        if self.arrFreq[0] != intervals[0]:
            raise ValueError("Intervals must start at self.arrFreq[0]")

        if self.arrFreq[-1] != intervals[-1]:
            raise ValueError("Intervals must end at self.arrFreq[-1]")

        if not np.allclose(intervals, np.sort(intervals)):
            raise ValueError("Interval bounds must be sorted")

        if intervals.shape[0] != data.shape[2]:
            raise ValueError("intervals.shape[0] != data.shape[2]")

        if self.arrFreq.shape[0] == 1:
            raise ValueError("Not possible in the single freq case")

        self._subBandIntervals = intervals
        self._subBandData = data
        self._subBandAssignment = None
        self._activeSubBands = None
        self._arrFourierData = None

    @property
    def subBandIntervals(self) -> np.ndarray:
        """Defining array of the stationary subbands

        See :py:obj:`eadf.eadf.EADF.defineSubBands`.

        .. note::

          This property is read only.

        Returns
        -------
        np.ndarray
            Defining array of the stationary subbands

        """
        return self._subBandIntervals

    @property
    def subBandAssignment(self) -> np.ndarray:
        """Assigment of the operating frequencies to stationary subbands

        This property returns a list of indices, which tells us which
        operating frequency is in which subband.
        See :py:obj:`eadf.eadf.EADF.subBandIntervals` and
        :py:obj:`eadf.eadf.EADF.operatingFrequencies`

        .. note::

          This property is cacheable.

        Returns
        -------
        np.ndarray
            Description of returned object.

        """
        if self._subBandAssignment is None:
            self._subBandAssignment = self._calcSubBandAssignment()
        return self._subBandAssignment

    def _calcSubBandAssignment(self) -> np.ndarray:
        """Assing operating frequencies to their stationary subbands

        This function iterates through the currently set operating
        frequencies and assigns them to the respective subband, just by
        simple search for the right interval. This way we can avoid
        calculating the antenna response several times for a single
        stationary subband.

        Returns
        -------
        np.ndarray
            Indexing array.

        """
        res = np.zeros_like(self.operatingFrequencies, dtype=np.int)
        for ii, ff in enumerate(self.operatingFrequencies):
            if ff in self.subBandIntervals:
                res[ii] = np.arange(self.subBandIntervals.shape[0])[
                    self.subBandIntervals == ff
                ][0]
            else:
                res[ii] = np.arange(self.subBandIntervals.shape[0])[
                    self.subBandIntervals < ff
                ][-1]

        return res

    @property
    def activeSubBands(self) -> np.ndarray:
        """Actually used stationary subbands

        If the operation frequencies are such that some stationary subbands do
        not contain any operation frequency, we can save some computation time.

        .. todo::

          Make use of this!

        Returns
        -------
        np.ndarray
            indices of active subbands.

        """
        if self._activeSubBands is None:
            (
                self._activeSubBands,
                self._subBandMapping,
            ) = self._calcActiveSubBands()

        return self._activeSubBands

    def _calcActiveSubBands(self) -> np.ndarray:
        # go through all possible subbands and check if any operation
        # frequencies are assigned to it.
        active = np.array(
            [
                bb
                for bb in range(len(self._subBandIntervals))
                if bb in self._subBandAssignment
            ]
        )

        mapping = np.zeros(len(active), dtype=np.int)
        for ii, aa in active:
            mapping[aa] = ii

        return active, mapping

    @property
    def operatingFrequencies(self) -> np.ndarray:
        """Frequencies where we whish to evaluate the pattern

        .. note::

          Setting this triggers the recalculation of all data used for
          calculating the pattern.

        .. note::

          This property is writeable.

        Returns
        -------
        np.ndarray
            Description of returned object.

        """
        return self._operatingFrequencies

    @operatingFrequencies.setter
    def operatingFrequencies(self, operatingFrequencies: np.ndarray,) -> None:
        if self.arrFreq.shape[0] == 1:
            raise ValueError("Not possible in the single freq case")

        if np.min(operatingFrequencies) < self.arrFreq[0]:
            raise ValueError("Unallowed Frequency, too small")

        if np.max(operatingFrequencies) > self.arrFreq[-1]:
            raise ValueError("Unallowed Frequency, too large")

        self._operatingFrequencies = np.sort(operatingFrequencies)

        if self.subBandIntervals is None:
            # if this is true, there are no subbands. In this case, we do a
            # spline interpolation of the fourier transformed input data
            # in order to generate the pattern data at given frequencies
            self._arrFourierData = None
            self._arrIndAziCompress = None
            self._arrIndCoEleCompress = None
            self._arrDataCalc = None
        else:
            # if this is false, we have defined subbands so that we need
            # to reevaluate the assignment of the operating frequencies
            # to their respective subbands.
            self._subBandAssignment = None
            self._activeSubBands = None

    @property
    def frequencySplines(self) -> np.ndarray:
        """Frequency Interpolation Splines

        These splines are used to get the pattern data for the specified
        :py:obj:`eadf.eadf.EADF.operatingFrequencies`. Since these may or may
        not be resided on the grid used during measurement of the array
        we use these splines to interpolate this data.

        .. note::

          This property is cacheable.

        Returns
        -------
        np.ndarray
            CoEle x Azi x Pol x Elem ndarray of Splines

        """
        if self._frequencySplines is None:
            self._frequencySplines = splineInterpolateFrequency(
                self._arrFreq, self._arrInputFourierData
            )

        return self._frequencySplines

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)

    def save(self, path: str) -> None:
        """Save the object to disk in a serialized way

        .. note::
            This is not safe! Make sure the requirements for pickling are met.
            Among these are different CPU architectures, Python versions,
            Numpy versions and so forth.

            However we at least check the eadf package version when reading
            back from disk.

        Parameters
        ----------
        path : str
            Path to write to
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str) -> object:
        """Load the Class from Serialized Data

        .. note::
            This is not safe! Make sure the requirements for pickling are met.
            Among these are different CPU architectures, Python versions,
            Numpy versions and so forth.

            However we at least check the eadf package version when reading
            back from disk and issue a warning if the versions don't match.
            then you are on your own!

        Parameters
        ----------
        path : str
            Path to load from

        Returns
        -------
        object
            The EADF object
        """
        with open(path, "rb") as file:
            res = pickle.load(file)

        from . import __version__

        if res.version != __version__:
            logging.warning(
                "eadf.load: loaded object does not match current version."
            )
        return res

    @property
    def version(self) -> None:
        """Return the version of the EADF package used to create the object

        This is important, if we pickle an EADF object and recreate it from
        disk with a possible API break between two versions of the package.
        Right now we only use the property to issue a warning to the user
        when the versions dont match when reading an object from disk.
        """
        return self._version

    def __init__(
        self,
        arrData: np.ndarray,
        arrCoEle: np.ndarray,
        arrAzi: np.ndarray,
        arrFreq: np.ndarray,
        arrPos: np.ndarray,
        **options
    ) -> None:
        """Initialize an EADF Object

        Here we assume that the input data is given in the internal data
        format already. If you have antenna data, which is not in the
        internat data format, we advice you to use one of the importers,
        or implement your own.

        In direction of co-elevation, we assume that both the north and the
        south pole were sampled, where the first sample represents the north
        pole and the last one the south pole. So arrCoEle must run from 0 to
        pi. In azimuth direction, we truncate the last
        sample, if we detect in arrAzi that both the first and last sample
        match. Both arrays have to contain values that are evenly spaced and
        ascending in value.

        Parameters
        ----------
        arrData : np.ndarray
            Co-Ele x Azi x Freq x Pol x Element
        arrCoEle : np.ndarray
            Co-elevation sampling positions in radians, both poles should be
            sampled
        arrAzi : np.ndarray
            Azimuth sampling positions in radians.
        arrFreq : np.ndarray
            Frequencies sampled at.
        arrPos : np.ndarray
            (3 x numElements) Positions of the single antenna elements.
            this is just for vizualisation purposes.

        """
        # Chek the input arguments and allow for readjustments of them
        arrData, arrCoEle, arrAzi, arrFreq, arrPos = _checkInputData(
            arrData, arrCoEle, arrAzi, arrFreq, arrPos, **options
        )

        arrData, self.unitRawData = arrData.m, arrData.u
        arrCoEle, self.unitCoEle = arrCoEle.m, arrCoEle.u
        arrAzi, self.unitAzi = arrAzi.m, arrAzi.u
        arrFreq, self.unitFreq = arrFreq.m, arrFreq.u
        arrPos, self.unitPos = arrPos.m, arrPos.u

        # truncate the beampattern data correctly
        # in azimuth we make sure that we did not sample the same angle twice
        if np.allclose(
            np.mod(arrAzi[0] + 2 * np.pi, 2 * np.pi),
            np.mod(arrAzi[-1], 2 * np.pi),
        ):
            arrAziTrunc = np.arange(arrAzi.shape[0] - 1)
        else:
            arrAziTrunc = np.arange(arrAzi.shape[0])

        # do the flipping and shifting in elevation and azimuth
        self._arrRawData = symmetrizeData(arrData[:, arrAziTrunc])

        # extract some meta data from the input
        self._arrPos = np.copy(arrPos)
        self._numElements = self._arrPos.shape[1]
        self._numCoEleInit = 2 * arrCoEle.shape[0] - 2
        self._numAziInit = arrAziTrunc.shape[0]
        self._arrCoEle = np.copy(arrCoEle.flatten())
        self._arrIndCoEleCompress = None
        self._arrAzi = np.copy(arrAzi.flatten()[arrAziTrunc])
        self._arrIndAziCompress = None
        self._arrFreq = np.copy(arrFreq.flatten())

        (
            self._arrInputFourierData,
            self._muCoEle,
            self._muAzi,
        ) = sampledToFourier(self._arrRawData[:, arrAziTrunc])

        self._muCoEleCalc = None
        self._muAziCalc = None
        self._arrDataCalc = None

        # first we use the measurement data that defines the array
        self._arrFourierData = self._arrInputFourierData

        self._frequencySplines = None
        self._operatingFrequencies = np.copy(self._arrFreq)
        self._subBandIntervals = None
        self._subBandAssignment = None
        self._activeSubBands = None

        # initialize some properties with defaults values
        self._blockSize = 128

        # we do no compression by default
        self.compressionFactor = 1.0

        # this is set in core.backend.lowMemory
        self._lowMemory = lowMemory

        # this is set in core.backend.dType
        self._dType = dtype

        # set the version
        self._version = __version__
