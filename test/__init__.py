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

import unittest
from contextlib import contextmanager


class TestCase(unittest.TestCase):
    @contextmanager
    def assertRaisesWithMessage(self, exception, message=None):
        """
        Allows to check if a piece of code raises the proper exception and
        also checks if the exception message is right.
        """
        with self.assertRaises(exception) as context:
            yield context

        if message is not None:
            strException = str(context.exception)
            self.assertTrue(
                message in strException,
                msg="'%s' != '%s'" % (message, str(context.exception)),
            )
