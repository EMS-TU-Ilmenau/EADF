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
Mathematical Core Routines
--------------------------

In this submodule we place all the mathematical and general core routines
which are used throughout the package. These are not intended for
direct use, but are still documented in order to allow new developers who
are unfamiliar with the code base to get used to the internal structure.
"""

from typing import Callable
import pint

import numpy as np
from .backend import cDtype

__all__ = [
    "calcBlockSize",
    "inversePatternTransform",
    "inversePatternTransformLowMem",
    "evaluatePattern",
    "evaluateGradient",
    "evaluateHessian",
]

ureg = pint.UnitRegistry()


def calcBlockSize(
    muCoEle: np.ndarray,
    muAzi: np.ndarray,
    arrData: np.ndarray,
    blockSizeMax: int,
    lowMem: bool,
) -> int:
    """Calculate an optimized block size

    This function steadily increases the blockSize during the pattern
    transform in order to optimze it for execution time. We only use a very
    crude metric and a very naive measurement for execution time. However,
    it is a starting point.

    Parameters
    ----------
    muCoEle : np.ndarray
        co-elevation DFT frequencies
    muAzi : np.ndarray
        azimuth DFT frequencies
    arrData : np.ndarray
        Fourier coefficients of the array
    blockSizeMax : int
        Maximum block Size
    lowMem : bool
        Is it the low memory mode?

    Returns
    -------
    int
        block size

    """
    import timeit

    # this also the initial block size
    # we also increase it by this value as long as it decreases
    stepSize = 128
    runTimeOld = 2
    runTimeNew = 1
    blockSizeOld = 0

    # iterate until the computation time increases
    while (runTimeOld >= runTimeNew) and (blockSizeOld < blockSizeMax):
        blockSizeNew = blockSizeOld + stepSize
        n = blockSizeOld + 2 * blockSizeNew
        if lowMem:

            azi = np.random.randn(n)
            coele = np.random.randn(n)

            def funCoEle(arrCoEle):
                return (np.exp(np.outer(1j * muCoEle, arrCoEle))).astype(
                    cDtype, order="C"
                )

            def funAzi(arrAzi):
                return (np.exp(np.outer(1j * muAzi, arrAzi))).astype(
                    cDtype, order="F"
                )

            def funBench():
                inversePatternTransformLowMem(
                    coele, azi, funCoEle, funAzi, arrData, blockSizeNew
                )

        else:
            ele = (np.exp(np.outer(1j * muCoEle, np.random.randn(n)))).astype(
                cDtype, order="C"
            )
            azi = (np.exp(np.outer(1j * muAzi, np.random.randn(n)))).astype(
                cDtype, order="F"
            )

            def funBench():
                inversePatternTransform(ele, azi, arrData, blockSizeNew)

        runTimeNew = (
            np.mean(np.asarray(timeit.repeat(funBench, repeat=5, number=10)))
            / n
        )

        # if the computation time increased for the first time, we
        # stop the iteration and return the previous blockSizeOld
        if runTimeNew > runTimeOld:
            break
        else:
            runTimeOld = runTimeNew
            blockSizeOld = blockSizeNew
    return blockSizeOld


def inversePatternTransform(
    arrCoEle: np.ndarray,
    arrAzi: np.ndarray,
    arrData: np.ndarray,
    blockSize: int,
) -> np.ndarray:
    """Samples the Pattern by using the Fourier Coefficients

    This function does the heavy lifting in the EADF evaluation process.
    It is used to sample the beampattern and the derivative itself, by
    evaluating d_phi * Gamma * d_theta^t as stated in (6) in the EADF
    paper by Landmann and delGaldo. It broadcasts this product over
    the last three coordinates of the fourier data, so across all
    wave frequency bins, polarisations and array elements.

    By changing d_theta(arrCoEle) and d_phi (arrAzi) acordingly in the
    arguments one can calculate either the derivative or the pattern itself.

    Parameters
    ----------
    arrCoEle : np.ndarray
        array of fourier kernels in co-elevation direction
    arrAzi : np.ndarray
        array of fourier kernels in azimuth direction
    arrData : np.ndarray
        the Fourier coefficients to use
    blockSize : int
        number of angles to process at once

    Returns
    -------
    np.ndarray
        beam pattern values at arrCoEle, arrAzi
    """
    # equation (6) in EADF paper by landmann and del galdo
    # np.einsum(
    #    "ij...,ik,jk->k...", arrData, arrCoEle, arrAzi
    # )
    arrRes = np.zeros(
        (arrCoEle.shape[1], *arrData.shape[2:]), dtype=arrData.dtype
    )
    numBlocks = int(arrCoEle.shape[1] / blockSize)

    # iterate over the blocks
    for bb in range(numBlocks):
        # only create the views once.
        ae = arrCoEle[:, bb * blockSize : (bb + 1) * blockSize]
        aa = arrAzi[:, bb * blockSize : (bb + 1) * blockSize]
        for jj in range(arrData.shape[2]):
            for kk in range(arrData.shape[3]):
                for ll in range(arrData.shape[4]):
                    arrRes[
                        bb * blockSize : (bb + 1) * blockSize, jj, kk, ll
                    ] = np.sum(ae * arrData[:, :, jj, kk, ll].dot(aa), axis=0)

    if (numBlocks * blockSize) < arrCoEle.shape[1]:
        ae = arrCoEle[:, numBlocks * blockSize :]
        aa = arrAzi[:, numBlocks * blockSize :]
        # iterate over the rest
        for jj in range(arrData.shape[2]):
            for kk in range(arrData.shape[3]):
                for ll in range(arrData.shape[4]):
                    arrRes[numBlocks * blockSize :, jj, kk, ll] = np.sum(
                        ae * arrData[:, :, jj, kk, ll].dot(aa), axis=0,
                    )
    return arrRes


def inversePatternTransformLowMem(
    arrCoEle: np.ndarray,
    arrAzi: np.ndarray,
    funCoEle: Callable[[np.ndarray], np.ndarray],
    funAzi: Callable[[np.ndarray], np.ndarray],
    arrData: np.ndarray,
    blockSize: int,
) -> np.ndarray:
    """Samples the Pattern by using the Fourier Coefficients

    This function does the heavy lifting in the :py:obj:`EADF` evaluation
    process. It is used to sample the beampattern and the derivative itself, by
    evaluating d_phi * Gamma * d_theta^t as stated in (6) in the EADF
    paper by Landmann and delGaldo. It broadcasts this product over
    the last three coordinates of the fourier data, so across all
    polarisations, wave frequency bins and array elements.

    However, the matrices containing the complex exponentials are calculated
    block wise and on the fly.

    By changing d_theta(arrCoEle) and d_phi (arrAzi) acordingly in the
    arguments one can calculate either the derivative or the pattern itself.

    Parameters
    ----------
    arrCoEle : np.ndarray
        array of fourier kernels in co-elevation direction
    arrAzi : np.ndarray
        array of fourier kernels in azimuth direction
    funAzi : method
        function that generates transform matrix in azimuth direction
    funCoEle : method
        function that generates transform matrix in frequency direction
    arrData : np.ndarray
        the Fourier coefficients to use
    blockSize : int
        number of blocks to transform at once

    Returns
    -------
    np.ndarray
        beam pattern values at arrCoEle, arrAzi
    """
    # equation (6) in EADF paper by landmann and del galdo
    # np.einsum(
    #    "ij...,ik,jk->k...", arrData, arrCoEle, arrAzi
    # )
    res = np.empty(
        (arrCoEle.shape[0], *arrData.shape[2:]), dtype=arrData.dtype
    )
    numBlocks = int(arrCoEle.shape[0] / blockSize)

    # iterate over the blocks
    for bb in range(numBlocks):
        # only create the views once.
        ae = funCoEle(arrCoEle[bb * blockSize : (bb + 1) * blockSize])
        aa = funAzi(arrAzi[bb * blockSize : (bb + 1) * blockSize])
        for jj in range(arrData.shape[2]):
            for kk in range(arrData.shape[3]):
                for ll in range(arrData.shape[4]):
                    res[
                        bb * blockSize : (bb + 1) * blockSize, jj, kk, ll
                    ] = np.sum(ae * arrData[:, :, jj, kk, ll].dot(aa), axis=0)

    if (numBlocks * blockSize) < arrCoEle.shape[0]:
        # iterate over the rest
        ae = funCoEle(arrCoEle[numBlocks * blockSize :])
        aa = funAzi(arrAzi[numBlocks * blockSize :])

        for jj in range(arrData.shape[2]):
            for kk in range(arrData.shape[3]):
                for ll in range(arrData.shape[4]):
                    res[numBlocks * blockSize :, jj, kk, ll] = np.sum(
                        ae * arrData[:, :, jj, kk, ll].dot(aa), axis=0
                    )
    return res


def evaluatePattern(
    arrCoEle: np.ndarray,
    arrAzi: np.ndarray,
    muCoEle: np.ndarray,
    muAzi: np.ndarray,
    arrData: np.ndarray,
    blockSize: int,
    lowMem: bool,
) -> np.ndarray:
    """Sample the Beampattern at Arbitrary Angles

    Parameters
    ----------
    arrCoEle : np.ndarray
        co-elevation angles to sample at in radians
    arrAzi : np.ndarray
        azimuth angles to sample at in radians
    muCoEle : np.ndarray
        spatial frequency bins in co-elevation direction
    muAzi : np.ndarray
        spatial frequency bins in azimuth direction
    arrData : np.ndarray
        fourier coefficients
    blockSize : int
        number of angles / frequencies to process at once
    lowMem : bool
        should we save memory?

    Returns
    -------
    np.ndarray
        sampled values
    """
    if lowMem:

        def funCoEle(arrCoEle):
            # equation (7) in the EADF Paper
            return np.exp(np.outer(1j * muCoEle, arrCoEle))

        def funAzi(arrAzi):
            # equation (7) in the EADF Paper
            return np.exp(np.outer(1j * muAzi, arrAzi))

        return inversePatternTransformLowMem(
            arrCoEle, arrAzi, funCoEle, funAzi, arrData, blockSize
        )
    else:
        # equation (7) in the EADF Paper
        arrMultCoEle = np.exp(np.outer(1j * muCoEle, arrCoEle))
        arrMultAzi = np.exp(np.outer(1j * muAzi, arrAzi))

        return inversePatternTransform(
            arrMultCoEle, arrMultAzi, arrData, blockSize
        )


def evaluateGradient(
    arrCoEle: np.ndarray,
    arrAzi: np.ndarray,
    muCoEle: np.ndarray,
    muAzi: np.ndarray,
    arrData: np.ndarray,
    blockSize: int,
    lowMem: bool,
) -> np.ndarray:
    """Sample the Beampattern Gradients at Arbitrary Angles

    Parameters
    ----------
    arrCoEle : np.ndarray
        co-elevation angles to sample at in radians
    arrAzi : np.ndarray
        azimuth angles to sample at in radians
    muCoEle : np.ndarray
        spatial frequency bins in co-elevation direction
    muAzi : np.ndarray
        spatial frequency bins in azimuth direction
    arrData : np.ndarray
        fourier coefficients
    blockSize : int
        number of angles / frequencies to process at once
    lowMem : bool
        should we save memory?

    Returns
    -------
        np.ndarray
    """
    if lowMem:

        def funCoEle(arrCoEle):
            # equation (7) in the EADF Paper
            return np.exp(np.outer(1j * muCoEle, arrCoEle))

        def funAzi(arrAzi):
            # equation (7) in the EADF Paper
            return np.exp(np.outer(1j * muAzi, arrAzi))

        def funDerivCoEle(arrCoEle):
            # equation (8) in the EADF Paper
            return np.multiply(
                np.pi * 1j * muCoEle,
                np.exp(np.outer(1j * muCoEle, arrCoEle)).T,
            ).T

        def funDerivAzi(arrAzi):
            # equation (8) in the EADF Paper
            return np.multiply(
                np.pi * 1j * muAzi, np.exp(np.outer(1j * muAzi, arrAzi)).T
            ).T

        return np.stack(
            (
                inversePatternTransformLowMem(
                    arrCoEle,
                    arrCoEle,
                    funDerivCoEle,
                    funAzi,
                    arrData,
                    blockSize,
                ),
                inversePatternTransformLowMem(
                    arrCoEle, arrAzi, funCoEle, funDerivAzi, arrData, blockSize
                ),
            ),
            axis=-1,
        )
    else:
        # equation (7) in the EADF Paper
        arrMultCoEle = np.exp(np.outer(1j * muCoEle, arrCoEle))
        arrMultAzi = np.exp(np.outer(1j * muAzi, arrAzi))

        # equation (8) in the EADF Paper
        arrMultCoEleDeriv = np.multiply(1j * muCoEle, arrMultCoEle.T).T
        arrMultAziDeriv = np.multiply(1j * muAzi, arrMultAzi.T).T

        # build up array of gradient by calling the pattern transform
        # twice and then stacking them along a new last dimension
        return np.stack(
            (
                inversePatternTransform(
                    arrMultCoEleDeriv, arrMultAzi, arrData, blockSize
                ),
                inversePatternTransform(
                    arrMultCoEle, arrMultAziDeriv, arrData, blockSize
                ),
            ),
            axis=-1,
        )


def evaluateHessian(
    arrCoEle: np.ndarray,
    arrAzi: np.ndarray,
    muCoEle: np.ndarray,
    muAzi: np.ndarray,
    arrData: np.ndarray,
    blockSize: int,
    lowMem: bool,
) -> np.ndarray:
    """Sample the Beampattern Hessian Matrix at Arbitrary Angles

    Parameters
    ----------
    arrCoEle : np.ndarray
        co-elevation angles to sample at in radians
    arrAzi : np.ndarray
        azimuth angles to sample at in radians
    muCoEle : np.ndarray
        spatial frequency bins in co-elevation direction
    muAzi : np.ndarray
        spatial frequency bins in azimuth direction
    arrData : np.ndarray
        fourier coefficients
    blockSize : int
        number of angles to process at once
    lowMem : bool
        should we save memory?

    Returns
    -------
        np.ndarray
    """
    if lowMem:

        def funCoEle(arrCoEle):
            # equation (7) in the EADF Paper
            return np.exp(np.outer(1j * muCoEle, arrCoEle))

        def funAzi(arrAzi):
            # equation (7) in the EADF Paper
            return np.exp(np.outer(1j * muAzi, arrAzi))

        def funDerivCoEle(arrCoEle):
            # equation (8) in the EADF Paper
            return np.multiply(
                1j * muCoEle, np.exp(np.outer(1j * muCoEle, arrCoEle)).T,
            ).T

        def funDerivAzi(arrAzi):
            # equation (8) in the EADF Paper
            return np.multiply(
                1j * muAzi, np.exp(np.outer(1j * muAzi, arrAzi)).T
            ).T

        def funDerivDerivCoEle(arrCoEle):
            # another derivative taken in (8) in the EADF paper
            return np.multiply(
                -(muCoEle ** 2), np.exp(np.outer(1j * muCoEle, arrCoEle)).T
            ).T

        def funDerivDerivAzi(arrAzi):
            # another derivative taken in (8) in the EADF paper
            return np.multiply(
                -(muAzi ** 2), np.exp(np.outer(1j * muAzi, arrAzi)).T
            ).T

        d11 = inversePatternTransformLowMem(
            arrCoEle, arrCoEle, funDerivDerivCoEle, funAzi, arrData, blockSize
        )
        d22 = inversePatternTransformLowMem(
            arrCoEle, arrAzi, funCoEle, funDerivDerivAzi, arrData, blockSize
        )
        d12 = inversePatternTransformLowMem(
            arrCoEle, arrAzi, funDerivCoEle, funDerivAzi, arrData, blockSize
        )

    else:
        # equation (7) in the EADF Paper
        arrMultCoEle = np.exp(np.outer(1j * muCoEle, arrCoEle))
        arrMultAzi = np.exp(np.outer(1j * muAzi, arrAzi))

        # equation (8) in the EADF Paper
        arrMultCoEleDeriv = np.multiply(1j * muCoEle, arrMultCoEle.T).T
        arrMultAziDeriv = np.multiply(1j * muAzi, arrMultAzi.T).T

        # another derivative taken in (8) in the EADF paper
        arrMultCoEleDerivDeriv = np.multiply(-(muCoEle ** 2), arrMultCoEle.T).T
        arrMultAziDerivDeriv = np.multiply(-(muAzi ** 2), arrMultAzi.T).T

        # build up array of gradient by calling the pattern transform
        # twice and then stacking them along a new last dimension
        d11 = inversePatternTransform(
            arrMultCoEleDerivDeriv, arrMultAzi, arrData, blockSize
        )
        d22 = inversePatternTransform(
            arrMultCoEle, arrMultAziDerivDeriv, arrData, blockSize
        )
        d12 = inversePatternTransform(
            arrMultCoEleDeriv, arrMultAziDeriv, arrData, blockSize
        )

    # this should return
    #     |d11|d12|
    # H = |---|---|
    #     |d12|d22|
    return np.stack(
        (np.stack((d11, d12.conj()), axis=-1), np.stack((d12, d22), axis=-1)),
        axis=-1,
    )
