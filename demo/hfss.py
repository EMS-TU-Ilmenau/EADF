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
Import Simulation Data from HFSS
--------------------------------

This demo outlines how you should feed data into our Ansys HFSS importer
in order to work with your fresh designs in Python. It mainly serves the
purpose to explain how one should export the data from HFSS correctly such that
the provided importer can deal with it.

Antenna design
^^^^^^^^^^^^^^

.. note::
  The objective of this design is to illustrate the steps of obtaining the
  required data for EADF pakage. Therefore, the model serves for this purpose
  only and it is not a proper antenna design for mmWave. The parameteres are
  ideal and no practical constraints are taken into account. However, the steps
  for obtaining the data are valid for any actual antenna design.


Procedure
^^^^^^^^^

We design a dual polarized printed dipole antenna for operating frequequcy of,
fo = 30 GHz with minimum -10dB bandwidth of 6 GHz.

#. Create a FR-4 substrate with dimensions (x, y, z) = (10, 5, 0.4) mm^3
   dielectric constant = 4.4, dielectric loss tangent = 0.02, postion the
   substrate at (x = -10/2, y = -5/2, z = 0)
#. Create a printed dipole using rectangular with dimensions of each arm as
   (width, length) = (0.5, 1.3) mm^2, The gap between the arms, g = 0.5 mm,
   Assign the sheet as Perefect E, Feed the dipole from the center using Lumped
   Port, Position the dipole at (x = -3, y = 0, z = 0) and then rotate it -45
   eg w.r.t z-axis (x-y plane), Design three more of these dipoles and placed
   them as follows:

   - (x = -3, y = 0, z = 0.4) and then rotate it 45 deg w.r.t z-axis
     (x-y plane)
   - (x = 3, y = 0, z = 0) and then rotate it -45 deg w.r.t z-axis (x-y plane)
   - (x = 3, y = 0, z = 0.4) and then rotate it 45 deg w.r.t z-axis (x-y plane)

#. Create an airbox with dimensions of (x, y, z) = (16, 8, 5.4) mm^3,
   Position it at (x = -16/2, y = -8/2, z = -2.5), Set the material as vacuum
#. Activate all the Ports, Assign 1 as the magnitude and 0 for Phase for
   all the four ports
#. Add a Solution Setup
   - Set the Frequency to 30 GHz
   - Add a Frequency Sweep
   - Choose Sweep Type as Fast
   - Choose Distribution as Linear Count with Points = 91

   .. note::
      Since this is a demo file Fast Sweep Type is chosen, otherwise the user
      should use the desired setup for their actual design.

#. Run the Simulation
#. Check S-Parameters and Radiation Pattern to make sure everything is done
   correctly
#. Create a Radiation for Far Field and set the Infinite Sphere as (0 deg < Phi
   < 360 deg) and (0 deg < Theta < 180 deg)
#. Results -> Create Far Fields Report -> Data Table. When the Report dialog
   pops up follow the steps:

   - Trace tab: Solution = Sweep; Category = rE; Quantity  = rE Phi
     and rE Theta; Function  = re and im
   - Families tab: Variables = Freq; Use frequencies of interest; Leave the
     rest of the setting unchanged and press on New Report; The created file is
     the required data

#. The result corresponding to a specific port could be obtained by assigning
   its magnitude to 1 and 0 for the remaining ports. Then, performing the
   revious step
#. Export the data as .csv file

.. figure:: _static/demo_hfss_1.png

.. figure:: _static/demo_hfss_2.png

.. figure:: _static/demo_hfss_3.png

>>> cd demo
>>> python hfss.py [list-of-csv-files]
"""


if __name__ == "__main__":
    import eadf
    import sys
    from numpy import *

    A = eadf.fromHFSS(*sys.argv[1:], sort_support=True, truncate_CoEle=(0, pi))
