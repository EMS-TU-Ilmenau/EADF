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
Input Data Vizualization
------------------------

This demo can be a starting point, if you first want to have a qick
but good look on your data before feeding it into the EADF for processing.

We create a plot, where you can slide through the excitation frequencies
and toggle the two polarisations and the respective field quantities
at the antenna ports.

>>> cd demo
>>> python inputviz.py

This then should yield something like the following.

.. figure:: _static/demo_inputviz.png

  Output for some exemplatory measurement data.
"""


import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RadioButtons


def loadData(path: str) -> tuple:
    r"""
    This function serves as a simple importer to get the data from disk
    into the format that is used by the EADF. Here you are on your own,
    since your data might be rearranged differently.

    The tuple that is returned should contain the beam pattern,
    angular values and frequency values.
    """
    data = scipy.io.loadmat(path)

    # this is just shit brought to you by matlab
    data["pattern"][0][0][0][0][0][1].flatten()
    arrCoEle = (
        xp.radians(data["pattern"][0][0][0][0][0][1].flatten()) + xp.pi / 2
    )
    arrAzi = xp.radians(data["pattern"][0][0][0][0][1][1].flatten())
    arrFreq = data["pattern"][0][0][0][0][4][1]
    arrPattern = data["pattern"][0][0][1]

    # bring the data in the EADF data format
    arrPattern = xp.swapaxes(arrPattern, 2, 4)

    return (arrPattern, arrCoEle, arrAzi, arrFreq)


def genFigure(arrPattern, arrCoEle, arrAzi, arrFreq) -> None:
    r"""
    This function should not need much modification. First, we
    setup the GUI, plot the initial data slice and finally
    create some GUI events that trigger the redrawing of the figure
    with the respective data slice.
    """
    # some mpl magic nobody gets
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.margins(x=0)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # make the sliders and radio buttons real
    axColor = "lightgoldenrodyellow"
    axFreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axColor)
    axElem = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axColor)
    axPol = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axColor)
    axE = plt.axes([0.025, 0.7, 0.15, 0.15], facecolor=axColor)

    # generate a slider for the excitation frequencies and
    # for the elements
    sldFreq = Slider(
        axFreq, "Freq", 0, arrPattern.shape[2] - 1, valinit=0, valfmt="%i"
    )
    sldElem = Slider(
        axElem, "Element", 0, arrPattern.shape[-1] - 1, valinit=0, valfmt="%i"
    )

    # we also throw in a radio button to switch between the polariations
    rdPol = RadioButtons(axPol, ("0", "1"), active=0)
    rdE = RadioButtons(axE, ("0", "1"), active=0)

    # this allows to switch between the two field quantities.
    lstFun = [xp.real, xp.imag]

    # draw the initial image
    img = ax.imshow(
        xp.real(arrPattern[:, :, 0, 0, 0]),
        vmin=-xp.max(xp.abs(arrPattern)),
        vmax=xp.max(xp.abs(arrPattern)),
        extent=[arrAzi[0], arrAzi[-1], arrCoEle[0], arrCoEle[-1]],
    )

    # this function is triggered whenever something in the GUI changes
    def update(val):
        freq = sldFreq.val
        elem = sldElem.val
        pol = rdPol.value_selected

        # select real or imaginary part
        fun = lstFun[int(rdE.value_selected)]
        img.set_data(fun(arrPattern[:, :, int(freq), int(pol), int(elem)]))
        fig.canvas.draw_idle()

    # attach GUI events to the update function
    rdPol.on_clicked(update)
    rdE.on_clicked(update)
    sldFreq.on_changed(update)
    sldElem.on_changed(update)
    fig.colorbar(img, ax=ax)
    plt.show()


if __name__ == "__main__":
    arrPattern, arrCoEle, arrAzi, arrFreq = loadData("pattern_IZTR5509.mat")
    genFigure(arrPattern, arrCoEle, arrAzi, arrFreq)
