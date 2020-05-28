import numpy as np
import eadf
import timeit
from eadf.backend import xp

numEle, numAzi = 100, 100

array = eadf.generateURA(
    1,
    1,
    0.5,
    0.75,
    xp.array([[30e-3, 30e-3, 1e-3]]).T,
    xp.linspace(5e9, 6e9, 3),
)
array._lowMemory = False
arrayPerf = eadf.PerformanceEADF(array)

ele, azi = eadf.toGrid(*eadf.sampleAngles(numEle, numAzi))


def Ap():
    array.pattern(ele, azi)


def Bp():
    arrayPerf.pattern(ele, azi)


def Ag():
    array.gradient(ele, azi)


def Bg():
    arrayPerf.gradient(ele, azi)


def Ah():
    array.hessian(ele, azi)


def Bh():
    arrayPerf.hessian(ele, azi)


print(
    "normal EADF pattern: %f" % np.mean(timeit.repeat(Ap, number=5, repeat=5))
)
print(
    "perf. EADF pattern: %f" % np.mean(timeit.repeat(Bp, number=5, repeat=5))
)
print(
    "normal EADF gradient: %f" % np.mean(timeit.repeat(Ag, number=5, repeat=5))
)
print(
    "perf. EADF gradient: %f" % np.mean(timeit.repeat(Bg, number=5, repeat=5))
)
print(
    "normal EADF hessian: %f" % np.mean(timeit.repeat(Ah, number=5, repeat=5))
)
print(
    "perf. EADF hessian: %f" % np.mean(timeit.repeat(Bh, number=5, repeat=5))
)
