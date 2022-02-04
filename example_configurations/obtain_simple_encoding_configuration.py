#!/usr/bin/env python3

"""
This script parses a full encoding configuration (which defines the encoding for every single segment) and creates a
simplified configuration with an encoding per column. It does so by chosing the encoding with the most occurences per
column. This simplified configuration can be directly used with Hyrise's benchmark binaries and the parameter `-e`.
Currently only applicable for TPC-H.
"""

import json
import lzma
import sys

from pathlib import Path


column_names = {"customer": ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal",
                             "c_mktsegment", "c_comment"],
                "orders":   ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate",
                             "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
                "lineitem": ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice",
                             "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate",
                             "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"],
                "part":     ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container",
                             "p_retailsize", "p_comment"],
                "partsupp": ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"],
                "supplier": ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal",
                             "s_comment"],
                "nation":   ["n_nationkey", "n_name", "n_regionkey", "n_comment"],
                "region":   ["r_regionkey", "r_name", "r_comment"]}


def prettify(encoding_tuple, count):
  ret = encoding_tuple[0][1]
  if len(encoding_tuple) > 1 and encoding_tuple[1] is not None:
    ret += f" ({encoding_tuple[1][1]})"
  return ret + f": {count:8,d}x"


assert len(sys.argv) == 2, "Please provide path to full encoding configuration."
configuration_path = Path(sys.argv[1])
assert configuration_path.exists(), "Provided configuration does not exist."

with lzma.open(configuration_path) as json_input_file:
  json_bytes = json_input_file.read()
  file_content = json_bytes.decode('utf-8')
  full_configuration = json.loads(file_content)["configuration"]

assert "lineitem" in full_configuration, "Script currently only supports TPC-H configurations."
simple_config_custom = {}

for table_name, chunks in full_configuration.items():
  encoding_counts = {}
  for _, chunk_configs in chunks.items():
    for column_id, encoding_config in chunk_configs.items():
      if column_id not in encoding_counts:
        encoding_counts[column_id] = {}

      hashable = tuple(sorted(encoding_config.items()))
      encoding_counts[column_id][hashable] = encoding_counts[column_id].setdefault(hashable, 0) + 1

  for column_id, counts in encoding_counts.items():
    alternatives_sorted = sorted(encoding_counts[column_id], key=encoding_counts[column_id].get, reverse=True)
    if len(alternatives_sorted) > 1:
      pretty_alternatives = [prettify(k, v) for k, v in encoding_counts[column_id].items()]
      alternatives_str = ', '.join(pretty_alternatives)
      print(f"Picking {prettify(alternatives_sorted[0], encoding_counts[column_id][alternatives_sorted[0]]):>60} from: \t\t{alternatives_str}")

    if table_name not in simple_config_custom:
      simple_config_custom[table_name] = {}
    simple_config_custom[table_name][column_names[table_name][int(column_id)]] = {"encoding": alternatives_sorted[0][0][1]}
    if len(alternatives_sorted[0]) > 1 and alternatives_sorted[0][1] is not None:
      simple_config_custom[table_name][column_names[table_name][int(column_id)]]["compression"] = alternatives_sorted[0][1][1]

json_output_filename = f"simple__{configuration_path.name.replace('.xz', '')}"
with open(json_output_filename, "w") as json_output_file:
  # The default is not used, as the full configurations always cover the entire benchmark data set. However, Hyrise expects it.
  json_output_file.write(json.dumps({"default": {"encoding": "Dictionary"}, "custom": simple_config_custom}, indent=4))

print(f"Wrote simplified encoding configuration file {json_output_filename}")

