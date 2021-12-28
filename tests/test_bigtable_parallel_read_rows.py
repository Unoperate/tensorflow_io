# Copyright 2021 Google LLC
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

# disable module docstring for tests
# pylint: disable=C0114
# disable class docstring for tests
# pylint: disable=C0115

import os
import subprocess
import re
import datetime
from re import escape
from tensorflow_io.python.ops import core_ops
from tensorflow_io.python.ops.bigtable.bigtable_dataset_ops import BigtableClient
from tensorflow import test
from threading import Thread
from typing import List






class BigtableReadTest(test.TestCase):

    def test_read(self):

        os.environ["BIGTABLE_EMULATOR_HOST"] = "127.0.0.1:8086"

        print("test read started")
        print("create table")


        print("create client")

        client = BigtableClient("fake_project", "fake_instance")
        print("get table")


