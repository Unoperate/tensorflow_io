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
from re import escape
from tensorflow_io.python.ops import core_ops
from tensorflow_io.python.ops.bigtable.bigtable_dataset_ops import BigtableClient
import tensorflow_io.python.ops.bigtable.bigtable_row_range as row_range
import tensorflow_io.python.ops.bigtable.bigtable_row_set as row_set
import tensorflow as tf
from tensorflow import test
from threading import Thread
from typing import List

CBT_CLI_SEARCH_PATHS = [
    "google-cloud-sdk/bin/cbt",
    "/usr/local/google-cloud-sdk/bin/cbt",
    "/usr/bin/cbt",
    "cbt",
]

CBT_CLI_PATH_ENV_VAR = "CBT_CLI_PATH"


def _get_cbt_binary_path(env_var_name, search_paths, description):
    res = os.environ.get(env_var_name)
    if res is not None:
        if not os.path.isfile(res):
            raise OSError(
                f"{description} specified in the {env_var_name} "
                "environment variable does not exist"
            )
        return res
    for candidate in search_paths:
        if os.path.isfile(candidate):
            return candidate
    raise OSError(f"Could not find {description}")


def _get_cbt_cli_path():
    return _get_cbt_binary_path(CBT_CLI_PATH_ENV_VAR, CBT_CLI_SEARCH_PATHS, "cbt cli")



class BigtableEmulator:
    def __init__(self):
        print("starting BigtableEmulator")

        self._emulator_addr = "localhost:8086"

        print("emulator addr", self._emulator_addr)

    def get_addr(self):
        return self._emulator_addr

    def create_table(
        self, project_id, instance_id, table_id, column_families, splits=None
    ):
        cli_path = _get_cbt_cli_path()
        cmd = [
            cli_path,
            "-project",
            project_id,
            "-instance",
            instance_id,
            "createtable",
            table_id,
            "families=" + ",".join([f"{fam}:never" for fam in column_families]),
        ]
        if splits:
            cmd.append("splits=" + ",".join(splits))
        subprocess.check_output(cmd)

    def write_tensor(
        self,
        project_id,
        instance_id,
        table_id,
        tensor: tf.Tensor,
        rows: List[str],
        columns: List[str],
    ):
        assert len(tensor.shape) == 2
        assert len(rows) == tensor.shape[0]
        assert len(columns) == tensor.shape[1]
        cli_path = _get_cbt_cli_path()
        for i, row in enumerate(tensor):
            for j, value in enumerate(row):
                cmd = [
                    cli_path,
                    "-project",
                    project_id,
                    "-instance",
                    instance_id,
                    "set",
                    table_id,
                    rows[i],
                    f"{columns[j]}={value.numpy().decode()}",
                ]
                subprocess.check_output(cmd)

    def stop(self):
        self._emulator.terminate()
        self._output_reading_thread.join()
        self._emulator.stdout.close()
        self._emulator.wait()




class BigtableReadTest(test.TestCase):
    def setUp(self):
        self.emulator = BigtableEmulator()

    def tearDown(self):
        self.emulator.stop()

    def test_read(self):
        print("test read started")
        print("emulator running at ", self.emulator.get_addr())
        os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
        print("create table")
        self.emulator.create_table(
            "fake_project", "fake_instance", "test-table", ["fam1", "fam2"]
        )

        values = [[f"[{i,j}]" for j in range(2)] for i in range(20)]

        ten = tf.constant(values)

        print("create client")

        client = BigtableClient("fake_project", "fake_instance")
        print("get table")
        table = client.get_table("test-table")

        print("write tensor")
        self.emulator.write_tensor(
            "fake_project",
            "fake_instance",
            "test-table",
            ten,
            ["row" + str(i).rjust(3, "0") for i in range(20)],
            ["fam1:col1", "fam2:col2"],
        )

        print("read rows")

        for i, r in enumerate(
            table.read_rows(
                ["fam1:col1", "fam2:col2"],
                row_set=row_set.from_rows_or_ranges(row_range.empty()),
            )
        ):
            for j, c in enumerate(r):
                self.assertEqual(values[i][j], c.numpy().decode())
