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
import tensorflow_io.python.ops.bigtable.bigtable_row_range as row_range
import tensorflow_io.python.ops.bigtable.bigtable_row_set as row_set
from google.auth.credentials import AnonymousCredentials
from google.cloud.bigtable import Client
from google.cloud.bigtable import column_family
import tensorflow as tf
from tensorflow import test
from threading import Thread
from typing import List


class BigtableEmulator:
    def __init__(self, project_id="fake_project", instance_id="fake_instance"):
        print("starting BigtableEmulator")

        os.environ["BIGTABLE_EMULATOR_HOST"] = "127.0.0.1:8086"

        self._client = Client(
            project=project_id, credentials=AnonymousCredentials(), admin=True
        )
        self._instance = self._client.instance(instance_id)

    def create_table(
        self, table_id, column_families=["cf1"]
    ):
        assert len(column_families) > 0

        table = self._instance.table(table_id)

        if table.exists():
            table.delete()

            table = self._instance.table(table_id)

        column_families = dict()
        for fam in column_families:
            max_versions_rule = column_family.MaxVersionsGCRule(2)
            column_families[fam] = max_versions_rule

        table.create(column_families=column_families)

    def write_tensor(
        self,
        table_id,
        tensor: tf.Tensor,
        rows: List[str],
        columns: List[str],
    ):
        assert len(tensor.shape) == 2
        assert len(rows) == tensor.shape[0]
        assert len(columns) == tensor.shape[1]

        table = self._instance.table(table_id)
        assert table.exists()

        rows = []
        for i, tensor_row in enumerate(tensor):
            row_key = "row" + str(i).rjust(3, "0")
            row = table.direct_row(row_key)
            for j, value in enumerate(tensor_row):
                family,column = columns[j].split(":")
                row.set_cell(
                    family, column, value.numpy(), timestamp=datetime.datetime.utcnow()
                )
            rows.append(row)
        table.mutate_rows(rows)



    def stop(self):
        self._client.close()




class BigtableReadTest(test.TestCase):

    def test_read(self):

        self.emulator = BigtableEmulator(
            "fake_project",
            "fake_instance")

        print("test read started")
        print("create table")
        self.emulator.create_table(
            "test_read", ["fam1", "fam2"]
        )

        values = [[f"[{i,j}]" for j in range(2)] for i in range(20)]

        ten = tf.constant(values)

        print("create client")

        client = BigtableClient("fake_project", "fake_instance")
        print("get table")
        table = client.get_table("test_read")

        self.emulator.stop()

