import tensorflow_io as tfio
from tensorflow_io import bigtable as bt

_=[print(x) for x in dir(bt)]

c = bt.BigtableClient("test-project", "test-instance")
t = c.get_table("t1")
row_s = row_set.from_rows_or_ranges(row_range.closed_range("row000", "row009"))

read_rows = [
    r for r in t.read_rows(["cf1:c1"], row_set=row_s)
]
print(read_rows)

