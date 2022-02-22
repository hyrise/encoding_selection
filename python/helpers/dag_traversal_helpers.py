#!/usr/bin/env python3

"""
This file should probably be renamed (or better be rewritten alltogether). Previously, we have traversed the query DAG
in order to determine if an operator's input was sequential or not. But this part is now already done in the Hyrise
plugins for performance reasons. So this is more a dag_helper.py
"""

import numpy as np
import pandas as pd
import multiprocess

import datetime
import os
import random
import sys
import time


from pathlib import Path


def add_meta_data_to_two_tables_measurements(df, meta_data):
    # Using left joins here, as there are joins against temporary columns (NULL on table name and column name)
    left_joined = pd.merge(df, meta_data, how="left", left_on=['LEFT_TABLE_NAME', 'LEFT_COLUMN_NAME'],
                      right_on=['TABLE_NAME', 'COLUMN_NAME'], suffixes=('_LEFT', ''))
    both_joined = pd.merge(left_joined, meta_data, how="left", left_on=['RIGHT_TABLE_NAME', 'RIGHT_COLUMN_NAME'],
                      right_on=['TABLE_NAME', 'COLUMN_NAME'], suffixes=('', '_RIGHT'))
    # several columns of the left-part join do not include the suffix _LEFT as there has not been a duplicate at this time.

    return both_joined


def add_meta_data_to_measurements(df, meta_data):
    # Using a left join here. Materialized accesses to columns (e.g., a column materialized in a projection) have both
    # the table and column names as NULL.
    joined = pd.merge(df, meta_data, how="left", on=['TABLE_NAME', 'COLUMN_NAME'])

    return joined


def load_and_process_calibration_files(calibration_folder, verbose = False):
    benchmark_folders = []

    print(calibration_folder)

    for dirpath, dirnames, _ in os.walk(calibration_folder):
        if not dirnames:
            if 'processed_for' in dirpath:
                continue

            benchmark_folders.append(dirpath)

    all_workload_operators = {}

    operator_files = {"JOIN": "joins.csv", "AGGREGATE": "aggregates.csv", "TABLE_SCAN": "table_scans.csv",
                      "PROJECTION": "projections.csv", "GET_TABLE": "get_tables.csv", "MISC": "misc_operators.csv"}

    dtype = {'COLUMN_NAME': 'string', 'TABLE_NAME': 'string'}

    for benchmark_folder in sorted(benchmark_folders):
        benchmark_folder_stem = Path(benchmark_folder).stem
        workload_operators = pd.DataFrame()

        for file in os.listdir(benchmark_folder):
            if file in operator_files.values():
                # Loading with explicit string columns for table and column name (they can be fully empty for TPC-C,
                # where one join column is a temporary column) and telling pandas to interpret the empty string as NULL
                data_characteristics = pd.read_csv(os.path.join(benchmark_folder, "..", "data_characteristics.csv"), dtype=dtype, keep_default_na=False)
                table_meta_data = pd.read_csv(os.path.join(benchmark_folder, "meta_data_table.csv"), dtype=dtype, keep_default_na=False)
                column_meta_data = pd.read_csv(os.path.join(benchmark_folder, "meta_data_column.csv"), dtype=dtype, keep_default_na=False)
                segment_meta_data = pd.read_csv(os.path.join(benchmark_folder, "meta_data_segment.csv"), dtype=dtype, keep_default_na=False)
                
                segment_meta_data = segment_meta_data.query('CHUNK_ID == 0')  # data of first chunk is sufficient
                meta_data_table_column = pd.merge(table_meta_data, column_meta_data, how="inner", on=['TABLE_NAME'])
                meta_data_table_segment = pd.merge(meta_data_table_column, segment_meta_data, how="inner", on=['TABLE_NAME', 'COLUMN_NAME'])
                meta_data = pd.merge(meta_data_table_segment, data_characteristics, how="inner", on=['TABLE_NAME', 'COLUMN_ID', 'CHUNK_ID'])
                data = pd.read_csv(os.path.join(benchmark_folder, file))
                operator_str = [k for k, v in operator_files.items() if v == file][0]

                if operator_str == "MISC":
                    data['OPERATOR_TYPE'] = data['OPERATOR'].str.upper()
                else:
                    data['OPERATOR_TYPE'] = operator_str

                if 'INPUT_SHUFFLEDNESS' not in data.columns:
                    data['INPUT_SHUFFLEDNESS'] = 0.0

                if len(data) > 0:  # TPC-C might have empty files for short runs with slow encodings
                    operator = data.iloc[0]['OPERATOR_TYPE']
                    if operator in ['JOIN', 'TABLE_SCAN']:
                        data = add_meta_data_to_two_tables_measurements(data, meta_data)
                    elif operator in ['PROJECTION', 'AGGREGATE']:
                        data = add_meta_data_to_measurements(data, meta_data)            

                    workload_operators = pd.concat([workload_operators, data])

        if len(workload_operators) == 0:
            continue

        workload_operators["is_temporary_column"] = np.where((workload_operators["TABLE_NAME"].isnull()) & (workload_operators["COLUMN_NAME"].isnull()), 1.0, 0.0)

        print(f'{datetime.datetime.now()} - Processing operator data for "{benchmark_folder_stem}".')
        process_dag(workload_operators, multiprocess.cpu_count() - 2, verbose)

        if verbose and 'dictionary_fwi' in benchmark_folder:
            print(f"{benchmark_folder} >>")
            operator_sums = workload_operators.groupby(['OPERATOR_TYPE', 'QUERY_HASH', 'OPERATOR_HASH']).agg({'RUNTIME_NS':['max']}).reset_index(level=1).groupby(['OPERATOR_TYPE']).sum()
            overall_runtime = operator_sums.RUNTIME_NS.sum()
            covered_share = 0.0
            calibrated_operators = ['TABLE_SCAN', 'JOIN', 'PROJECTION', 'AGGREGATE']
            print(f'\tRuntime share of calibrated operators (overall {overall_runtime[0]/1000/1000/1000:,.4f} s): ', end='')
            for _, row in operator_sums.iterrows():
                if row.name in calibrated_operators:
                    operator_share = row['RUNTIME_NS'].sum() / overall_runtime
                    covered_share += operator_share[0]
                    print(f"{row.name} {operator_share[0]:.2%}     ", end='')
            print("")

            for _, row in operator_sums.iterrows():
                print(f"""\t{row.name.capitalize().replace("_", " "):>20} (ms): {operator_sums.loc[[row.name]]["RUNTIME_NS"]["max"][0]/1000/1000:10,.2f}""")

        all_workload_operators[benchmark_folder] = workload_operators

    total_operator_lines = 0
    total_operators = 0
    total_queries = 0
    for workload in all_workload_operators.values():
        total_operator_lines += len(workload)
        total_operators += len(workload["OPERATOR_HASH"].unique())
        total_queries += len(workload["QUERY_HASH"].unique())

    if verbose: print(f'Total number of parsed operator CSV lines: {total_operators:,} operators in {total_operator_lines:,} lines ({total_queries:,} queries).')

    return all_workload_operators


def process_dag(df, process_count = int(multiprocess.cpu_count() / 2), verbose = False):
    # Handle NULL values, introduced due to left join. NULLable scans can be a scan on a "COUNT(*)" column
    df['NULLABLE'] = np.where(df.is_temporary_column == 1.0, 1.0, df.NULLABLE)  # assume NULLability for temporary columns
    df['NULLABLE'] = np.where(df.COLUMN_NAME == "COUNT(*)", 0.0, df.NULLABLE)

    # that's very rough and also at least in some instances wrong (e.g., MIN(ps_supplycost) join).
    df['DATA_TYPE'] = np.where(df.is_temporary_column == 1.0, "int", df.DATA_TYPE)
    df['DATA_TYPE_RIGHT'] = np.where(df.is_temporary_column == 1.0, "int", df.DATA_TYPE_RIGHT)
    df['DATA_TYPE'] = np.where(df.COLUMN_NAME == "COUNT(*)", "int", df.DATA_TYPE)
    df['DATA_TYPE_RIGHT'] = np.where(df.COLUMN_NAME == "COUNT(*)", "int", df.DATA_TYPE_RIGHT)

    # distinct count for temporary columns is hard to estimate, so we just take the middle between 1 and full
    # distinctiveness (2^16-1)
    df['DISTINCT_VALUE_COUNT'] = np.where((df.is_temporary_column == 1.0) | (df.COLUMN_NAME == "COUNT(*)"), 2**15, df.DISTINCT_VALUE_COUNT)
    df['DISTINCT_VALUE_COUNT_RIGHT'] = np.where((df.is_temporary_column == 1.0) | (df.COLUMN_NAME == "COUNT(*)"), 2**15, df.DISTINCT_VALUE_COUNT_RIGHT)
    # sanity check
    df['DISTINCT_VALUE_COUNT'].fillna(2**15, inplace=True)
    df['DISTINCT_VALUE_COUNT_RIGHT'].fillna(2**15, inplace=True)

    df['ENCODING_TYPE'] = np.where((df.is_temporary_column == 1.0) | (df.COLUMN_NAME == "COUNT(*)"), "Unencoded", df.ENCODING_TYPE)

    df['INPUT_SHUFFLEDNESS'] = np.where(df.is_temporary_column == 1.0, 0.0, df.INPUT_SHUFFLEDNESS)
    df['INPUT_SHUFFLEDNESS'] = np.where(df.COLUMN_NAME == "COUNT(*)", 0.0, df.INPUT_SHUFFLEDNESS)


def write_results_to_disk(all_workload_operators, folder, verbose = False):
    os.makedirs(folder, exist_ok = True)

    operators = set()
    for workload_operators in all_workload_operators.values():
        operators.update(workload_operators['OPERATOR_TYPE'].unique())

    for operator in operators:
        df = pd.DataFrame()
        for workload_operators in all_workload_operators.values():
            operator_instances = workload_operators.query(f'OPERATOR_TYPE == "{operator}"')
            df = pd.concat([df, operator_instances])

        # Sanity check. We rely on operator instances being identifiable by their hash. The combination QUERY_HASH and
        # OPERATOR_HASH always identify a single operator instance.
        grouping_1 = df.groupby(["QUERY_HASH", "OPERATOR_HASH"])
        grouping_2 = df.groupby(["QUERY_HASH", "OPERATOR_HASH", "DESCRIPTION", "RUNTIME_NS"])
        if grouping_1.ngroups != grouping_2.ngroups:
            print(f"Unexpected collisions for query-operator hash combinations ({grouping_2.ngroups} unique instances, {grouping_1.ngroups} hashes")
            group_sizes = grouping_2.size()
            print(group_sizes[group_sizes > 1])

        df.to_csv(os.path.join(folder, f'{operator.lower()}.csv.bz2'))
        if verbose: print(f'Wrote data for {operator.lower():^15s} operator to file: {len(df): >10,} rows.')

