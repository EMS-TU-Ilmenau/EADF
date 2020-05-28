# -*- coding: UTF8 -*-
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
Modelling Antenna Arrays with the EADF
======================================

Motivation
----------

Geometry-based MIMO channel modelling and a high-resolution parameter
estimation are applications in which a precise description of the radiation
pattern of the antenna arrays is required. In this package we implement an
efficient representation of the polarimetric antenna response, which we refer
to as the Effective Aperture Distribution Function (EADF). High-resolution
parameter estimation are applications in which this reduced description permits
us to efficiently interpolate the beam pattern to gather the antenna response
for an arbitrary direction in azimuth and elevation. Moreover, the EADF
provides a continuous description of the array manifold and its derivatives
with respect to azimuth and elevation. The latter is valuable for the
performance evaluation of an antenna array as well as for gradient-based
parameter estimation techniques.

References
----------

*Full 3D Antenna Pattern Interpolation Using Fourier Transform
Based Wavefield Modelling*; S. Haefner, R. Mueller, R. S. Thomae;
WSA 2016; 20th International ITG Workshop on Smart Antennas;
Munich, Germany; pp. 1-8

*Impact of Incomplete and Inaccurate Data Models on High Resolution Parameter
Estimation in Multidimensional Channel Sounding*, M. Landmann, M. Käske
and R.S. Thomä; IEEE Trans. on Antennas and Propagation, vol. 60, no 2,
February 2012, pp. 557-573

*Efficient antenna description for MIMO channel modelling and estimation*,
M. Landmann, G. Del Galdo; 7th European Conference on Wireless Technology;
Amsterdam; 2004; `IEEE Link <https://ieeexplore.ieee.org/document/1394809>`_

*Geometry-based Channel Modeling for Multi-User MIMO Systems and
Applications*, G. Del Galdo; Dissertation, Research Reports from the
Communications Research Laboratory at TU Ilmenau; Ilmenau; 2007
`Download1 <https://www.db-thueringen.de/servlets/MCRFileNodeServlet/
dbt_derivate_00014957/ilm1-2007000136.pdf>`_

*Limitations of Experimental Channel Characterisation*, M. Landmann;
Dissertation; Ilmenau; 2008 `Download2 <https://www.db-thueringen.de/servlets
/MCRFileNodeServlet/dbt_derivate_00015967/ilm1-2008000090.pdf>`_

*cupy* `Cupy Documentation
<https://docs-cupy.chainer.org/en/stable/>`_

*Intel Python* `Intelpython Documentation
<https://software.intel.com/en-us/distribution-for-python>`_
"""


__version__ = "0.5"


from .arrays import *
from .auxiliary import *
from .backend import *
from .eadf import *
from .importers import *
from .preprocess import *
from .sphericalharm import *
