#!/usr/bin/env python3

import datetime
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
import re
import sys
import time

from pathlib import Path
from timeit import default_timer as timer

sys.path.append("..")
from helpers import encoding_selection_constants


def add_features(df):
    df['input_is_reference_table'] = np.where(df.eval("COLUMN_TYPE == 'REFERENCE'"), 1.0, 0.0)

    # preparing the vector compression acronym. We later want to have a single column like "DictionaryBitPacking"
    replacements = {"FixedW": "_fwi", "BitPac": "_bp"}
    def add_segment_encoding(suffix):
        df[f'VECTOR_COMPRESSION_TYPE{suffix}'].fillna('', inplace=True)
        df[f'VECTOR_COMPRESSION_TYPE_ACRONYM{suffix}'] = [replacements[x[:6]] if x[:6] in replacements else '' for x in df[f'VECTOR_COMPRESSION_TYPE{suffix}']]

        suffix_low = suffix.lower()
        # Unencoded is no encoding (thus NULL in segment meta data)
        df[f'ENCODING_TYPE{suffix}'].fillna('Unencoded', inplace=True)

        df[f'segment_encoding_spec{suffix_low}'] = df[f'ENCODING_TYPE{suffix}'].str.lower() + df[f'VECTOR_COMPRESSION_TYPE_ACRONYM{suffix}']
        # only neccesary until we fixed the size info creation
        df[f'segment_encoding_spec{suffix_low}'] = np.where(df[f'segment_encoding_spec{suffix_low}'] == 'lz4',
                                                            'lz4_bp', df[f'segment_encoding_spec{suffix_low}'])


    add_segment_encoding('')
    # for join and table scan DF, left and right columns have encodings (but only right is suffixed)
    if 'ENCODING_TYPE_RIGHT' in df.columns:
        add_segment_encoding('_RIGHT')
        # duplicate left columns
        df['segment_encoding_spec_left'] = df['segment_encoding_spec']
        df['left_input_is_reference_table'] = np.where(df.eval("LEFT_COLUMN_TYPE == 'REFERENCE'"), 1.0, 0.0)
        df['right_input_is_reference_table'] = np.where(df.eval("RIGHT_COLUMN_TYPE == 'REFERENCE'"), 1.0, 0.0)

    df['input_row_count_log'] = np.log(df['INPUT_ROW_COUNT'] + 1)
    df['input_row_count_log'].fillna(0.0, inplace=True)

    create_hot_one_log_input_table_sizes = False
    if create_hot_one_log_input_table_sizes:
        # Creating 1-0 features for log10-size of input table
        input_row_count_log = np.round(np.log10(df['INPUT_ROW_COUNT'] + 1)).astype(int)
        input_count_log_binary = np.eye(max(10, max(input_count_log) + 1))[input_count_log]

        for i in range(np.shape(input_count_log_binary)[1]):
            col_name = f'left_input_row_count_binary_log10__{i}'
            df[col_name] = input_count_log_binary[:,i].astype(int)
    
    df['execution_time_ms'] = df["RUNTIME_NS"].astype(float) / 1000 / 1000


#######
#######
#######     TABLE SCANS
#######
#######

def add_features_table_scans(df, verbose = False):
    # We want to differentiate between different scenarios: wildcard upfront (always quite expensive), wildcard in the
    # back (cheap, rewritten to efficient range predicates) and everything else.
    # We want to ensure that only once case matches. Not more.
    preceding_like_pattern = re.compile(r"TableScan Impl: ColumnLike .*'%.*")
    trailing_like_pattern = re.compile(r"TableScan Impl: ColumnLike .*'.+%'")
    double_like_pattern = re.compile(r"TableScan Impl: ColumnLike .*'.+%.+%")  # TODO: should be no longer necessary with the upcoming RE-improvements
    NOT_like_pattern = re.compile(r"TableScan Impl: ColumnLike .+ NOT LIKE '")

    marker_preceding_like_pattern = np.where(df['DESCRIPTION'].str.contains(preceding_like_pattern), 1.0, 0)
    marker_trailing_like_pattern = np.where(df['DESCRIPTION'].str.contains(trailing_like_pattern), 1.0, 0)
    marker_double_like_pattern = np.where(df['DESCRIPTION'].str.contains(double_like_pattern), 1.0, 0)
    marker_NOT_like_pattern = np.where(df['DESCRIPTION'].str.contains(NOT_like_pattern), 1.0, 0)

    # If there is a trailing match, don't count it if it's also a preceeding or double match. Because a trailing match
    # is so super cheap. To keep it simple, we do not count preceeding ones when it's also a double one.
    marker_trailing_like_pattern = marker_trailing_like_pattern * (marker_trailing_like_pattern - marker_preceding_like_pattern)
    marker_trailing_like_pattern = marker_trailing_like_pattern * (marker_trailing_like_pattern - marker_double_like_pattern)
    marker_preceding_like_pattern = marker_preceding_like_pattern * (marker_preceding_like_pattern - marker_double_like_pattern)

    marker_column_vs_column = np.where(df['DESCRIPTION'].str.contains(" ColumnVsColumn ", regex=False), 1.0, 0)
    marker_between = np.where(df['DESCRIPTION'].str.contains(" ColumnBetween ", regex=False), 1.0, 0)
    marker_expression_evaluator = np.where(df['DESCRIPTION'].str.contains(" ExpressionEvaluator ", regex=False), 1.0, 0)
    marker_subquery = np.where(df['DESCRIPTION'].str.contains(" SUBQUERY ", regex=False), 1.0, 0)

    # In case of an expression evaluator scan on a subquery results, we count it as an expression evaluation
    marker_subquery = marker_subquery * (marker_subquery - marker_expression_evaluator)

    marker_simple_predicate = 1 - (marker_preceding_like_pattern + marker_trailing_like_pattern + marker_double_like_pattern + marker_column_vs_column + marker_expression_evaluator + marker_subquery + marker_between)
    if max(marker_simple_predicate) > 1 or  min(marker_simple_predicate) < 0:
        df["marker_simple_predicate"] = marker_simple_predicate
        df["marker_preceding_like_pattern"] = marker_preceding_like_pattern
        df["marker_trailing_like_pattern"] = marker_trailing_like_pattern
        df["marker_double_like_pattern"] = marker_double_like_pattern
        df["marker_column_vs_column"] = marker_column_vs_column
        df["marker_between"] = marker_between
        df["marker_expression_evaluator"] = marker_expression_evaluator
        df["marker_subquery"] = marker_subquery
        # print(df[["DESCRIPTION"] + [col for col in df.columns if "marker" in col]].query("marker_simple_predicate < 0 or marker_simple_predicate > 1"))
    assert max(marker_simple_predicate) <= 1
    assert min(marker_simple_predicate) >= 0

    expression_evaluator_matches = np.where(df['DESCRIPTION'].str.contains("ExpressionEvaluator"), 1, 0)
    tmp_list_begins = df['DESCRIPTION'].str.find("IN (")
    list_begins = np.where(tmp_list_begins == -1, -1, tmp_list_begins + len("IN ("))
    list_ends = df['DESCRIPTION'].str.rfind(")")
    description_in_remainders = [A[B:C] for A, B, C in zip(df.DESCRIPTION, list_begins, list_ends)]
    comma_counts = np.array([s.count(",") for s in description_in_remainders])
    df['expression_evaluator_in_list_element_count'] = np.where(comma_counts == 0, 0, (comma_counts + 1) * df.INPUT_ROW_COUNT)

    df['expression_evaluator_case_count'] = np.where(expression_evaluator_matches, df['DESCRIPTION'].str.count(" CASE WHEN ") * df.INPUT_ROW_COUNT, 0)
    df['expression_evaluator_OR_count'] = np.where(expression_evaluator_matches, df['DESCRIPTION'].str.count(" OR ") * df.INPUT_ROW_COUNT, 0)

    # Finding the data type of temporary columns is currently not that easy. We thus manually adjust it whenever we find a SUBSTR function.
    df['DATA_TYPE'] = np.where(df['DESCRIPTION'].str.contains("SUBSTR"), "string", df.DATA_TYPE)

    # df['expected_qualifying_rows_distance_to_50_percent'] = abs((df['OUTPUT_ROW_COUNT'] / (df['INPUT_ROW_COUNT'] + 1)) - 0.5) * df.INPUT_ROW_COUNT
    # df['expected_qualifying_rows_below_50_percent'] = (df['expected_qualifying_rows'] < 0.5).astype(int) * df.INPUT_ROW_COUNT
    
    df['rows_read_from_data_table'] = (1 - df.input_is_reference_table) * df.INPUT_ROW_COUNT
    df['rows_read_from_seq_ref_table'] = df.input_is_reference_table * (1 - df.INPUT_SHUFFLEDNESS) * df.INPUT_ROW_COUNT
    df['rows_read_from_nonseq_ref_table'] = df.input_is_reference_table * df.INPUT_SHUFFLEDNESS * df.INPUT_ROW_COUNT

    df['rows_per_chunk'] = df.INPUT_ROW_COUNT / df.INPUT_CHUNK_COUNT
    df['rows_per_chunk'].fillna(0.0, inplace=True)

    df['row_scans_skipped'] = np.where(df.INPUT_CHUNK_COUNT > 0, ((df.PRUNED_CHUNK_COUNT +  df.ALL_ROWS_MATCHED_COUNT) / df.INPUT_CHUNK_COUNT) * df.INPUT_ROW_COUNT, 0)
    df['row_scans_sorted'] = np.where(df.INPUT_CHUNK_COUNT > 0, (df.BINARY_SEARCH_COUNT / df.INPUT_CHUNK_COUNT) * df.INPUT_ROW_COUNT, 0)

    dictionary_encoding_types = ["dictionary_bp", "fixedstringdictionary_bp", "dictionary_fwi", "fixedstringdictionary_fwi"]
    dictionary_mask = np.isin(df[f"segment_encoding_spec_left"], dictionary_encoding_types)

    # Chunks are only skipped with dictionary encoding; pruning based on min/max happens in the optimizer and is not
    # visible within the table scan.
    df['chunks_per_row'] = ((dictionary_mask * (df.INPUT_CHUNK_COUNT - df.PRUNED_CHUNK_COUNT)) +
                            ((1 - dictionary_mask) * df.INPUT_CHUNK_COUNT)) / \
                           ((dictionary_mask * (df.INPUT_ROW_COUNT - (df.PRUNED_CHUNK_COUNT / df.INPUT_CHUNK_COUNT) * df.INPUT_ROW_COUNT)) +
                            ((1 - dictionary_mask) * df.INPUT_ROW_COUNT))
    df['chunks_per_row'] = np.nan_to_num(df['chunks_per_row'])

    input_is_data_table = (1 - df.input_is_reference_table)
    input_is_seq_ref_table = df.input_is_reference_table * (1 - df.INPUT_SHUFFLEDNESS)
    input_is_nonseq_ref_table = df.input_is_reference_table * df.INPUT_SHUFFLEDNESS
    columns_to_append = {}
    for data_type in encoding_selection_constants.DATA_TYPES:
        data_type_marker = np.where(data_type == df.DATA_TYPE, 1.0, 0.0)

        for encoding in encoding_selection_constants.ENCODINGS:
            for side in ["left", "right"]:
                encoding_marker = np.where(encoding == df[f"segment_encoding_spec_{side}"], 1.0, 0.0)

                rows_marker_data_type_encoding = data_type_marker * encoding_marker * \
                                                 ((dictionary_mask * (df.INPUT_ROW_COUNT - ((df.PRUNED_CHUNK_COUNT +  df.ALL_ROWS_MATCHED_COUNT) / df.INPUT_CHUNK_COUNT) * df.INPUT_ROW_COUNT)) +
                                                 ((1 - dictionary_mask) * df.INPUT_ROW_COUNT))
                rows_marker_data_type_encoding = np.nan_to_num(rows_marker_data_type_encoding)

                # We always want to track the left column of a table scan, but the right only for columnvscolumn scans
                # (NULLs for right table and column can mean there is no right column OR it's a temporary column).
                df['record_rows'] = 1.0
                df['record_rows'] = np.where(side == "right", marker_column_vs_column, df['record_rows'])

                for predicate_name, marker in [("preceding_like_pattern", marker_preceding_like_pattern),
                                               ("trailing_like_pattern", marker_trailing_like_pattern),
                                               ("double_like_pattern", marker_double_like_pattern),
                                               ("column_vs_column", marker_column_vs_column),
                                               ("between", marker_between),
                                               ("expression_evaluator", marker_expression_evaluator),
                                               ("subquery", marker_subquery),
                                               ("simple_predicate", marker_simple_predicate)]:
                    if side == "right" and predicate_name != "column_vs_column":
                        # Only column-vs-column have a right side (subquery is a single value as it's uncorrelated)
                        continue

                    if "like" in predicate_name:
                        if data_type != "string":
                            continue

                        # handle "like" and "not like"
                        for like_predicate_name, like_marker in [(predicate_name, (1 - marker_NOT_like_pattern)),
                                                                 (predicate_name.replace("_like_", "_notlike_"), marker_NOT_like_pattern)]:
                            columns_to_append[f"{side}_cells_read__{like_predicate_name}__{data_type}_{encoding}_from_data_table"] = like_marker * marker * input_is_data_table * \
                                                                                                                      rows_marker_data_type_encoding * df['record_rows']
                            columns_to_append[f"{side}_cells_read__{like_predicate_name}__{data_type}_{encoding}_from_seq_ref_table"] = like_marker * marker * input_is_seq_ref_table * \
                                                                                                                         rows_marker_data_type_encoding * df['record_rows']
                            columns_to_append[f"{side}_cells_read__{like_predicate_name}__{data_type}_{encoding}_from_nonseq_ref_table"] = like_marker * marker * input_is_nonseq_ref_table * \
                                                                                                                            rows_marker_data_type_encoding * df['record_rows']
                    else:
                        columns_to_append[f"{side}_cells_read__{predicate_name}__{data_type}_{encoding}_from_data_table"] = marker * input_is_data_table * rows_marker_data_type_encoding * df['record_rows']
                        columns_to_append[f"{side}_cells_read__{predicate_name}__{data_type}_{encoding}_from_seq_ref_table"] = marker * input_is_seq_ref_table * rows_marker_data_type_encoding * df['record_rows']
                        columns_to_append[f"{side}_cells_read__{predicate_name}__{data_type}_{encoding}_from_nonseq_ref_table"] = marker * input_is_nonseq_ref_table * rows_marker_data_type_encoding * df['record_rows']

    tmp_df = pd.DataFrame(columns_to_append)
    rows_before_concat = len(df)
    assert rows_before_concat == len(tmp_df)
    df = pd.concat([df, tmp_df], axis=1)
    assert len(df) == rows_before_concat, "pd.concat unexpectedly yielded new rows."

    return df


def get_table_scan_data(read_folder):
    file = os.path.join(read_folder, 'table_scan.csv.bz2')
    table_scans = pd.read_csv(file, low_memory=False)

    return table_scans


def featurize_table_scans(df, process_count = int(multiprocessing.cpu_count() / 2), verbose = False):
    add_features(df)
    table_scans = add_features_table_scans(df)

    df_prediction_projection = table_scans[encoding_selection_constants.PREDICTION_COLUMNS['table_scan']]
    assert len(df_prediction_projection[df_prediction_projection.isnull().any(axis=1)]) == 0

    return table_scans


def write_table_scan_data(df, write_folder):
    file = os.path.join(write_folder, 'table_scan_prepared.csv.bz2')
    df.to_csv(file, sep = ',', index = False)


#######
#######
#######     AGGREGATES
#######
#######

def add_features_aggregations(df):
    # Hyrise has an optimized path using std::array for certain group by widths
    single_column_group_by = np.where(df.GROUP_BY_COLUMN_COUNT == 1, 1.0, 0.0)
    two_columns_group_by = np.where(df.GROUP_BY_COLUMN_COUNT == 2, 1.0, 0.0)
    non_opt_group_by = np.where(df.GROUP_BY_COLUMN_COUNT > 2, 1.0, 0.0)

    columns_to_append = {}

    # We know for every column whether it is a group by column (largely accessed sequentially) or an aggregate column
    # (accessed mostly randomized). Before grouping per operator instance, we'll add this knowledge and later use
    # SUM() on the generated columns.
    for data_type in encoding_selection_constants.DATA_TYPES:
        data_type_marker = np.where(data_type == df.DATA_TYPE, 1.0, 0.0)
        for encoding in encoding_selection_constants.ENCODINGS:
            groupby_sequential_from_data_table = (1 - df.input_is_reference_table)
            groupby_sequential_from_ref_table = df.input_is_reference_table * (1 - df.INPUT_SHUFFLEDNESS)
            groupby_nonseq_from_ref_table = df.input_is_reference_table * df.INPUT_SHUFFLEDNESS

            for name, vector in [("single_column", single_column_group_by),
                                 ("two_columns", two_columns_group_by),
                                 ("non_opt", non_opt_group_by)]:
                # Grouping is sequential with a signifiant overhead for the hash table.
                columns_to_append[f'{data_type}_cells_read_in_{name}_groupby_columns_with_{encoding}_from_data_table'] = np.where(df['segment_encoding_spec'] == encoding,
                                                                                                                   vector * data_type_marker * df.IS_GROUP_BY_COLUMN * \
                                                                                                                   groupby_sequential_from_data_table * df.INPUT_ROW_COUNT,
                                                                                                                   0.0)
                columns_to_append[f'{data_type}_cells_read_in_{name}_groupby_columns_with_{encoding}_from_seq_ref_table'] = np.where(df['segment_encoding_spec'] == encoding,
                                                                                                                      vector * data_type_marker * df.IS_GROUP_BY_COLUMN * \
                                                                                                                      groupby_sequential_from_ref_table * df.INPUT_ROW_COUNT,
                                                                                                                      0.0)
                columns_to_append[f'{data_type}_cells_read_in_{name}_groupby_columns_with_{encoding}_nonseq'] = np.where(df['segment_encoding_spec'] == encoding,
                                                                                                          vector * data_type_marker * df.IS_GROUP_BY_COLUMN * \
                                                                                                          groupby_nonseq_from_ref_table * df.INPUT_ROW_COUNT,
                                                                                                          0.0)

            # no matter what in the input data is, if we first group, the aggregates are most certainly random accesses
            agg_is_grouped = np.where(df.GROUP_BY_COLUMN_COUNT > 0, 1.0, 0.0)
            sequential_from_data_table = (1 - df.input_is_reference_table) * (1 - agg_is_grouped)
            sequential_from_ref_table = df.input_is_reference_table * (1 - agg_is_grouped) * (1 - df.INPUT_SHUFFLEDNESS)
            tmp_nonseq_from_ref_table = df.input_is_reference_table * (1 - agg_is_grouped) * df.INPUT_SHUFFLEDNESS
            tmp_df = pd.DataFrame({"agg_is_grouped": agg_is_grouped, "tmp_nonseq_from_ref_table": tmp_nonseq_from_ref_table})
            non_sequential = tmp_df[["agg_is_grouped", "tmp_nonseq_from_ref_table"]].max(axis=1)

            # aggregations can be sequential without grouping, otherwise they are non-sequential
            for agg_name, vector in [("countstar", df.IS_COUNT_STAR),
                                     ("calculation", (1 - df.IS_COUNT_STAR))]:
                columns_to_append[f'{data_type}_cells_read_in_{agg_name}_aggregate_columns_with_{encoding}_from_data_table'] = np.where(df['segment_encoding_spec'] == encoding,
                                                                                                              vector * data_type_marker * (1 - df.IS_GROUP_BY_COLUMN) * \
                                                                                                              sequential_from_data_table * df.INPUT_ROW_COUNT,
                                                                                                              0.0)
                columns_to_append[f'{data_type}_cells_read_in_{agg_name}_aggregate_columns_with_{encoding}_from_seq_ref_table'] = np.where(df['segment_encoding_spec'] == encoding,
                                                                                                                 vector * data_type_marker * (1 - df.IS_GROUP_BY_COLUMN) * \
                                                                                                                 sequential_from_ref_table * df.INPUT_ROW_COUNT,
                                                                                                                 0.0)
                columns_to_append[f'{data_type}_cells_read_in_{agg_name}_aggregate_columns_with_{encoding}_nonseq'] = np.where(df['segment_encoding_spec'] == encoding,
                                                                                                     vector * data_type_marker * (1 - df.IS_GROUP_BY_COLUMN) * \
                                                                                                     non_sequential * df.INPUT_ROW_COUNT,
                                                                                                     0.0)

    # determine how many aggregates are calculated per row in the aggregation phase. The more, the worse the
    # performance due to the constant switching.
    non_consecutive_aggregations = np.power(df.AGGREGATE_COLUMN_COUNT - 1, 1.5)
    columns_to_append["non_consecutive_aggregation_calculations"] = non_consecutive_aggregations * df.INPUT_ROW_COUNT

    tmp_df = pd.DataFrame(columns_to_append)
    rows_before_concat = len(df)
    assert rows_before_concat == len(tmp_df)
    df = pd.concat([df, tmp_df], axis=1)
    assert len(df) == rows_before_concat, "pd.concat yielded new rows. I don't want that."

    grouped = df.groupby(['QUERY_HASH', 'OPERATOR_HASH'])

    # grouping and using max for "boring" features such as input size which is equal for all rows of an aggregation operator
    grouped_max = grouped[['QUERY_HASH', # used to later obtain the initial query
                           'OPERATOR_HASH',
                           'INPUT_ROW_COUNT',
                           'INPUT_CHUNK_COUNT',
                           'OUTPUT_ROW_COUNT',
                           'OUTPUT_CHUNK_COUNT',
                           'GROUP_BY_COLUMN_COUNT',
                           'AGGREGATE_COLUMN_COUNT',
                           'IS_COUNT_STAR',
                           'DESCRIPTION',
                           'input_row_count_log',
                           'execution_time_ms']].max()
    # For some columns, we take a simple mean/sum of all columns' properties.
    agg_dict = {'INPUT_SHUFFLEDNESS': 'mean',
                'input_is_reference_table': 'mean',
                'DISTINCT_VALUE_COUNT': 'mean'}
    for data_type in encoding_selection_constants.DATA_TYPES:
        for encoding in encoding_selection_constants.ENCODINGS:
            for name in ["single_column", "two_columns", "non_opt"]:
                agg_dict[f'{data_type}_cells_read_in_{name}_groupby_columns_with_{encoding}_from_data_table'] = 'sum'
                agg_dict[f'{data_type}_cells_read_in_{name}_groupby_columns_with_{encoding}_from_seq_ref_table'] = 'sum'
                agg_dict[f'{data_type}_cells_read_in_{name}_groupby_columns_with_{encoding}_nonseq'] = 'sum'

            for agg_name in ["countstar", "calculation"]:
                agg_dict[f'{data_type}_cells_read_in_{agg_name}_aggregate_columns_with_{encoding}_from_data_table'] = 'sum'
                agg_dict[f'{data_type}_cells_read_in_{agg_name}_aggregate_columns_with_{encoding}_from_seq_ref_table'] = 'sum'
                agg_dict[f'{data_type}_cells_read_in_{agg_name}_aggregate_columns_with_{encoding}_nonseq'] = 'sum'
    agg_dict["non_consecutive_aggregation_calculations"] = 'min'

    custom_column_aggs = grouped.agg(agg_dict)

    distinct_counts_agg = df.groupby(['QUERY_HASH', 'OPERATOR_HASH', 'IS_GROUP_BY_COLUMN']).DISTINCT_VALUE_COUNT.sum().unstack()

    data_types_agg = df.groupby(['QUERY_HASH', 'OPERATOR_HASH', 'DATA_TYPE']).DATA_TYPE.count().unstack()
    encodings_agg = df.groupby(['QUERY_HASH', 'OPERATOR_HASH', 'segment_encoding_spec']).segment_encoding_spec.count().unstack()

    result = pd.concat([grouped_max, custom_column_aggs, distinct_counts_agg, data_types_agg, encodings_agg], axis=1, sort=False).fillna(0)
    for data_type_to_add in ['float', 'int', 'string']:
        # add in case hasn't occurred in training data
        if data_type_to_add not in result.columns:
            result[data_type_to_add] = 0.0

    # COUNT() is an aggregate from my point of view. However, as we don't access data for it, we treat it differently.
    result["AGGREGATE_COLUMN_COUNT"] = result["AGGREGATE_COLUMN_COUNT"] - result["IS_COUNT_STAR"]

    result = result.rename({'float': 'float_cells_to_process',
                            'int': 'integer_cells_to_process',
                            'string': 'string_cells_to_process',
                            1.0: 'avg_distinct_values_groupby_columns',
                            0.0: 'avg_distinct_values_aggregate_columns',
                            'input_is_reference_table': 'cells_read_from_reference_table',
                            'DISTINCT_VALUE_COUNT': 'avg_distinct_value_count',
                            'GROUP_BY_COLUMN_COUNT': 'cells_read_for_grouping',
                            'AGGREGATE_COLUMN_COUNT': 'cells_read_for_aggregating'}, axis='columns')

    # JOB does never group
    for column in ['avg_distinct_values_groupby_columns', 'avg_distinct_values_aggregate_columns']:
        if column not in result.columns:
            result[column] = 0.0

    ####
    #### handle ANY() and DISTINCT()
    ####

    # several columns of interest are now multiplied by the input_row_count
    columns_to_multiply_by_row_count = ['float_cells_to_process',
                                        'integer_cells_to_process',
                                        'string_cells_to_process',
                                        'cells_read_from_reference_table',
                                        'cells_read_for_grouping',
                                        'cells_read_for_aggregating']
    for column_name in columns_to_multiply_by_row_count:
        result[column_name] = result[column_name] * result.INPUT_ROW_COUNT

    column_names = result.columns
    for encoding in encoding_selection_constants.ENCODINGS:
        if encoding in column_names:
            result[encoding] *= result.INPUT_ROW_COUNT
            result = result.rename(columns={encoding: f'cells_encoded_with_{encoding}'})
        else:
            result[f'cells_encoded_with_{encoding}'] = 0.0

    result["chunks_per_row"] = result.INPUT_CHUNK_COUNT / (result.INPUT_ROW_COUNT + 1)

    return result


def get_aggregate_data(read_folder):
    file = os.path.join(read_folder, 'aggregate.csv.bz2')
    aggregates = pd.read_csv(file, low_memory=False)
    return aggregates


def featurize_aggregates(df, process_count = int(multiprocessing.cpu_count() / 2), verbose = False):
    start = timer()
    add_features(df)
    aggregations = add_features_aggregations(df)

    df_prediction_projection = aggregations[encoding_selection_constants.PREDICTION_COLUMNS['aggregate']]
    assert len(df_prediction_projection[df_prediction_projection.isnull().any(axis=1)]) == 0
    assert len(df.groupby(['QUERY_HASH', 'OPERATOR_HASH'])) == len(aggregations)

    return aggregations


def write_aggregate_data(df, write_folder):
    file = os.path.join(write_folder, 'aggregate_prepared.csv.bz2')
    df.to_csv(file, sep = ',', index = False)


#######
#######
#######     JOINS
#######
#######

def process_hash_join_materialize_rows(df):
    # We yield two observations per join, one for each materialization.
    # We take as the fields from either left or right and simply switch the runtimes to include the fact that sides
    # could have been switched.
    def add_join_features(df, side):
        assert side in ["left", "right"]
        if side == "left":
            flip_side = "right"
        else:
            flip_side = "left"
        row_count_name = f"{side.upper()}_TABLE_ROW_COUNT"
        chunk_count_name = f"{side.upper()}_TABLE_CHUNK_COUNT"
        shuffledness_name = f"{side.upper()}_INPUT_SHUFFLEDNESS"

        distinct_name = "DISTINCT_VALUE_COUNT" if side == "left" else "DISTINCT_VALUE_COUNT_RIGHT"
        flip_distinct_name = "DISTINCT_VALUE_COUNT" if side == "right" else "DISTINCT_VALUE_COUNT_RIGHT"

        data_type_name = "DATA_TYPE" if side == "left" else "DATA_TYPE_RIGHT"
        flip_data_type_name = "DATA_TYPE" if side == "right" else "DATA_TYPE_RIGHT"

        is_reference_table = f"{side}_input_is_reference_table"
        segment_enc_name = f"segment_encoding_spec_{side}"

        df["TABLE_NAME"] = df[f"{side.upper()}_TABLE_NAME"]
        df["COLUMN_NAME"] = df[f"{side.upper()}_COLUMN_NAME"]
        df["DATA_TYPE"] = df[data_type_name]
        df["ROW_COUNT"] = df[f"{side.upper()}_TABLE_ROW_COUNT"]

        df["rows_per_chunk"] = np.where(df[chunk_count_name] > 0, df[row_count_name] / df[chunk_count_name], 0.0)
        df["chunks_per_row"] = df[f"{side.upper()}_TABLE_CHUNK_COUNT"] / (df[f"{side.upper()}_TABLE_ROW_COUNT"] + 1)
        radix_partition_count = 2**df["RADIX_BITS"]
        df["rows_per_radix_partition"] = df[f"{side.upper()}_TABLE_ROW_COUNT"] / radix_partition_count
        df["distinct_value_count"] = df[distinct_name]

        df["is_reference_table"] = df[f"{side}_input_is_reference_table"]
        df["shuffledness"] = df[f"{side.upper()}_INPUT_SHUFFLEDNESS"]

        df["materialize_side"] = side

        if side == "left":
            df["execution_time_ms"] = np.where(df.IS_FLIPPED, df.PROBE_SIDE_MATERIALIZING_NS, df.BUILD_SIDE_MATERIALIZING_NS)
        elif side == "right":
            df["execution_time_ms"] = np.where(df.IS_FLIPPED, df.BUILD_SIDE_MATERIALIZING_NS, df.PROBE_SIDE_MATERIALIZING_NS)
        else:
            Fail()

        df["execution_time_ms"] = (df["execution_time_ms"] * 1.0) / 1000 / 1000

        dictionary_encoding_types = ["dictionary_bp", "fixedstringdictionary_bp", "dictionary_fwi", "fixedstringdictionary_fwi"]
        dictionary_mask = np.isin(df[f"segment_encoding_spec_{side}"], dictionary_encoding_types)
        df["estimated_dictionary_searches"] = np.log2(df["distinct_value_count"] + 1) * dictionary_mask * df.ROW_COUNT

        columns_to_append = {}

        for data_type in encoding_selection_constants.DATA_TYPES:
            data_type_marker = np.where(data_type == df.DATA_TYPE, 1.0, 0.0)

            if side == "left":
                df[f"materialized_{data_type}_values"] = data_type_marker * np.where(df.IS_FLIPPED, df.PROBE_SIDE_MATERIALIZED_VALUE_COUNT, df.BUILD_SIDE_MATERIALIZED_VALUE_COUNT)
            elif side == "right":
                df[f"materialized_{data_type}_values"] = data_type_marker * np.where(df.IS_FLIPPED, df.BUILD_SIDE_MATERIALIZED_VALUE_COUNT, df.PROBE_SIDE_MATERIALIZED_VALUE_COUNT)

            for encoding in encoding_selection_constants.ENCODINGS:
                tmp_segment_encodings = df[f"segment_encoding_spec_{side}"]
                encoding_marker = np.where(encoding == tmp_segment_encodings, 1.0, 0.0)

                # In case the current size is larger, the other side has built up a bloom filter which is used in materialization.
                data_type_encoding_marker_rows = data_type_marker * encoding_marker * df.ROW_COUNT

                columns_to_append[f"cells_read__{data_type}_{encoding}__data_table"] = (1 - df.is_reference_table) * data_type_encoding_marker_rows
                columns_to_append[f"cells_read__{data_type}_{encoding}__seq_ref_table"] = (1 - df.shuffledness) * df[is_reference_table] * data_type_encoding_marker_rows
                columns_to_append[f"cells_read__{data_type}_{encoding}__nonseq_ref_table"] = df.shuffledness * df[is_reference_table] * data_type_encoding_marker_rows

                row_count_to_materialize = np.where(df[f"{side.upper()}_TABLE_ROW_COUNT"] > df[f"{flip_side.upper()}_TABLE_ROW_COUNT"],
                                                    df[f"materialized_{data_type}_values"], df.ROW_COUNT)
                data_type_encoding_marker_materialize_rows = data_type_marker * encoding_marker * row_count_to_materialize

                columns_to_append[f"cells_materialized__{data_type}_{encoding}__data_table"] = (1 - df.is_reference_table) * data_type_encoding_marker_materialize_rows 
                columns_to_append[f"cells_materialized__{data_type}_{encoding}__seq_ref_table"] = (1 - df.shuffledness) * df[is_reference_table] * data_type_encoding_marker_materialize_rows
                columns_to_append[f"cells_materialized__{data_type}_{encoding}__nonseq_ref_table"] = df.shuffledness * df[is_reference_table] * data_type_encoding_marker_materialize_rows


        tmp_df = pd.DataFrame(columns_to_append)
        rows_before_concat = len(df)
        assert rows_before_concat == len(tmp_df)
        df = pd.concat([df, tmp_df], axis=1)
        assert len(df) == rows_before_concat, "pd.concat yielded new rows."

        return df

    left = df[['QUERY_HASH',
               'OPERATOR_HASH',
               'LEFT_TABLE_NAME',
               'LEFT_COLUMN_NAME',
               'LEFT_TABLE_ROW_COUNT',
               'LEFT_TABLE_CHUNK_COUNT',
               'RIGHT_TABLE_NAME',
               'RIGHT_COLUMN_NAME',
               'RIGHT_TABLE_ROW_COUNT',
               'RIGHT_TABLE_CHUNK_COUNT',
               'RADIX_BITS',
               'PREDICATE_COUNT',
               'DATA_TYPE',
               'DATA_TYPE_RIGHT',
               'DISTINCT_VALUE_COUNT',
               'DISTINCT_VALUE_COUNT_RIGHT',
               'LEFT_INPUT_SHUFFLEDNESS',
               'RIGHT_INPUT_SHUFFLEDNESS',
               'left_input_is_reference_table',
               'right_input_is_reference_table',
               'IS_FLIPPED',
               'segment_encoding_spec_left',
               'segment_encoding_spec_right',
               'BUILD_SIDE_MATERIALIZING_NS',
               'PROBE_SIDE_MATERIALIZING_NS',
               'BUILD_SIDE_MATERIALIZED_VALUE_COUNT',
               'PROBE_SIDE_MATERIALIZED_VALUE_COUNT',
               'PRIMARY_PREDICATE']].copy()
    right = left.copy()

    left = add_join_features(left, 'left')
    right = add_join_features(right, 'right')

    combined = pd.concat([left, right])

    return combined


def process_hash_join_remainder_rows(df):
    power_value = 2.0

    df['build_rows_per_radix_partition'] = df.BUILD_SIDE_MATERIALIZED_VALUE_COUNT / (2**df.RADIX_BITS)
    df['probe_rows_per_radix_partition'] = df.PROBE_SIDE_MATERIALIZED_VALUE_COUNT / (2**df.RADIX_BITS)

    df['build_side_row_count_log'] = np.log(df.BUILD_SIDE_MATERIALIZED_VALUE_COUNT + 1)
    df['probe_side_row_count_log'] = np.log(df.PROBE_SIDE_MATERIALIZED_VALUE_COUNT + 1)

    for data_type in encoding_selection_constants.DATA_TYPES:
        ## Probe phase
        # Semi and anti joins use different hash maps without stored positions
        semi_anti_like = np.where((df.JOIN_MODE == "Semi") | (df.JOIN_MODE == "AntiNullAsFalse") | (df.JOIN_MODE == "AntiNullAsTrue"), 1.0, 0.0)
        non_semi_anti_inner_like = np.where((df.JOIN_MODE != "Semi") & (df.JOIN_MODE != "AntiNullAsFalse") & (df.JOIN_MODE != "AntiNullAsTrue") & (df.JOIN_MODE == "Inner"), 1.0, 0.0)
        remainder = np.where((df.JOIN_MODE != "Semi") & (df.JOIN_MODE != "AntiNullAsFalse") & (df.JOIN_MODE != "AntiNullAsTrue") & (df.JOIN_MODE == "Inner"), 1.0, 0.0)

        # Simple heuristic: if we have few distinct values, the probing and building will profit from the pos lists
        # being cache-resident.
        distinctness_factor = df.HASH_TABLES_DISTINCT_VALUE_COUNT / df.BUILD_SIDE_MATERIALIZED_VALUE_COUNT
        distinctness_factor = np.nan_to_num(distinctness_factor)
        if len(df) > 0:
            assert max(distinctness_factor) <= 1.0
            assert min(distinctness_factor) >= 0.0

        # Thought about using a sigmoid function, but we invest too much to have such large hash tables that we're even
        # out the of L3 cache. We still "somewhat" account for large hash tables.
        build_row_multiplier_tmp = 1 + (np.sqrt(distinctness_factor) / 5)
        # For semi/anti join, we only care about the distinct value count (no positions stored).
        build_row_multiplier = semi_anti_like + (1 - semi_anti_like) * build_row_multiplier_tmp
        data_type_match = np.where(data_type == df.DATA_TYPE, 1.0, 0.0)

        ## Cluster phase
        cluster_factor = np.where(df.RADIX_BITS == 0, 0.0, 1.5 - (1 / 2**df.RADIX_BITS))
        df[f'cells_to_cluster_{data_type}'] = data_type_match * (df.BUILD_SIDE_MATERIALIZED_VALUE_COUNT + df.PROBE_SIDE_MATERIALIZED_VALUE_COUNT) * cluster_factor

        ## Build phase
        # build costs increase slightly more than linear. Costs for semi/anti maps are lower as no positions need to be stored
        df[f'cells_for_semi_anti_build_phase_{data_type}'] = data_type_match * semi_anti_like * (2**df.RADIX_BITS) * df['build_rows_per_radix_partition']
        df[f'cells_for_non_semi_anti_build_phase_{data_type}'] = data_type_match * (1 - semi_anti_like) * (2**df.RADIX_BITS) * df['build_rows_per_radix_partition']

        ## Probe phase
        # Semi and anti joins use different hash maps without stored positions
        semi_anti_like = np.where((df.JOIN_MODE == "Semi") | (df.JOIN_MODE == "AntiNullAsFalse") | (df.JOIN_MODE == "AntiNullAsTrue"), 1.0, 0.0)
        non_semi_anti_inner_like = np.where(df.JOIN_MODE == "Inner", 1.0, 0.0)
        remainder = np.where((df.JOIN_MODE != "Semi") & (df.JOIN_MODE != "AntiNullAsFalse") & (df.JOIN_MODE != "AntiNullAsTrue") & (df.JOIN_MODE != "Inner"), 1.0, 0.0)

        df[f'rows_to_probe_semi_anti_like_{data_type}'] = data_type_match * semi_anti_like * df.PROBE_SIDE_MATERIALIZED_VALUE_COUNT * build_row_multiplier
        df[f'rows_to_probe_non_semi_anti_inner_like_{data_type}'] = data_type_match * non_semi_anti_inner_like * df.PROBE_SIDE_MATERIALIZED_VALUE_COUNT * build_row_multiplier
        df[f'rows_to_probe_remainer_{data_type}'] = data_type_match * remainder * df.PROBE_SIDE_MATERIALIZED_VALUE_COUNT * build_row_multiplier

    df['rows_probed_for_secondary_predicates'] = (df.PREDICATE_COUNT - 1) * df[["BUILD_SIDE_MATERIALIZED_VALUE_COUNT", "PROBE_SIDE_MATERIALIZED_VALUE_COUNT"]].max(axis=1)

    # very rough estimate
    df['join_complexity'] = df.PROBE_SIDE_MATERIALIZED_VALUE_COUNT * np.power(df.BUILD_SIDE_MATERIALIZED_VALUE_COUNT, 1.2)
    df['execution_time_ms'] = ((df.CLUSTERING_NS + df.BUILDING_NS + df.PROBING_NS + df.OUTPUT_WRITING_NS) * 1.0) / 1000 / 1000

    df['hash_table_position_count'] = df.HASH_TABLES_POSITION_COUNT
    df['output_row_count'] = df.OUTPUT_ROW_COUNT

    # Used to later simplify the joining
    df['materialize_side'] = "left"

    sizes_to_fill = df[["LEFT_TABLE_ROW_COUNT", "RIGHT_TABLE_ROW_COUNT"]].max(axis=1)

    return df


# Sort-merge joins happen only once in TPC-DS. Thus we implemented them just very briefly. The current model does not
# even consider encoding at all.
def process_sort_merge_joins(df):
    df['left_distinct_value_count'] = df.DISTINCT_VALUE_COUNT * df.LEFT_TABLE_CHUNK_COUNT
    df['right_distinct_value_count'] = df.DISTINCT_VALUE_COUNT * df.LEFT_TABLE_CHUNK_COUNT

    df['left_input_row_count_log'] = np.log(df.LEFT_TABLE_ROW_COUNT + 1)
    df['right_input_row_count_log'] = np.log(df.RIGHT_TABLE_ROW_COUNT + 1)

    is_not_inner = np.where(df.JOIN_MODE != "Inner", 1.0, 0.0)
    df['rows_joined_not_inner'] = is_not_inner * (df.RIGHT_TABLE_ROW_COUNT + df.LEFT_TABLE_ROW_COUNT)

    is_one_sided_outer_join = np.where((df.JOIN_MODE == "LeftOuter") | (df.JOIN_MODE == "RightOuter"), 1.0, 0.0)
    df['rows_joined_one_sided_outer_join'] = is_one_sided_outer_join * (df.RIGHT_TABLE_ROW_COUNT + df.LEFT_TABLE_ROW_COUNT)

    is_full_outer_join = np.where(df.JOIN_MODE == "FullOuter", 1.0, 0.0)
    df['rows_joined_full_outer_join'] = is_full_outer_join * (df.RIGHT_TABLE_ROW_COUNT + df.LEFT_TABLE_ROW_COUNT)

    df['rows_joined_non_equi'] = np.where(df.PRIMARY_PREDICATE != "=", df.LEFT_TABLE_ROW_COUNT + df.RIGHT_TABLE_ROW_COUNT, 0.0)
    df['rows_joined_for_secondary_predicates'] = np.where(df.PREDICATE_COUNT > 1, df.LEFT_TABLE_ROW_COUNT + df.RIGHT_TABLE_ROW_COUNT, 0.0)

    # very rough estimate
    df['left_sort_complexity'] = np.log2(df.LEFT_TABLE_ROW_COUNT + 1) * df.LEFT_TABLE_ROW_COUNT
    df['right_sort_complexity'] = np.log2(df.RIGHT_TABLE_ROW_COUNT + 1) * df.RIGHT_TABLE_ROW_COUNT

    sizes_to_fill = df[["LEFT_TABLE_ROW_COUNT", "RIGHT_TABLE_ROW_COUNT"]].max(axis=1)

    for join_mode in encoding_selection_constants.JOIN_MODES:
        df[f'join_mode_{join_mode}'] = np.where(join_mode == df.JOIN_MODE, sizes_to_fill, 0.0)

    for predicate_condition in encoding_selection_constants.PREDICATE_CONDITIONS:
        df[f'predicate_condition_{predicate_condition}'] = np.where(predicate_condition == df.PRIMARY_PREDICATE, sizes_to_fill, 0.0)

    # Used to later simplify the joining
    df['materialize_side'] = "left"

    return df


def get_join_data(read_folder):
    file = os.path.join(read_folder, 'join.csv.bz2')
    joins = pd.read_csv(file, low_memory=False)
    return joins


def featurize_joins(df, process_count = int(multiprocessing.cpu_count() / 2), verbose = False):
    add_features(df)

    # we will later copy data frame, so early projection helps
    df = df[['QUERY_HASH',
             'OPERATOR_HASH',
             'RADIX_BITS',
             'JOIN_MODE',
             'DATA_TYPE',
             'DATA_TYPE_RIGHT',
             'PRIMARY_PREDICATE',
             'PREDICATE_COUNT',
             'DISTINCT_VALUE_COUNT_RIGHT',
             'segment_encoding_spec_left',
             'segment_encoding_spec_right',
             'BUILD_SIDE_MATERIALIZING_NS',
             'PROBE_SIDE_MATERIALIZING_NS',
             'BUILD_SIDE_MATERIALIZED_VALUE_COUNT',
             'PROBE_SIDE_MATERIALIZED_VALUE_COUNT',
             'BUILDING_NS',
             'CLUSTERING_NS',
             'PROBING_NS',
             'OUTPUT_WRITING_NS',
             'IS_FLIPPED',
             'LEFT_TABLE_NAME',
             'LEFT_COLUMN_NAME',
             'LEFT_TABLE_ROW_COUNT',
             'LEFT_TABLE_CHUNK_COUNT',
             'RIGHT_TABLE_NAME',
             'RIGHT_COLUMN_NAME',
             'RIGHT_TABLE_ROW_COUNT',
             'RIGHT_TABLE_CHUNK_COUNT',
             'OUTPUT_ROW_COUNT',
             'DISTINCT_VALUE_COUNT',
             'INPUT_SHUFFLEDNESS',
             'LEFT_INPUT_SHUFFLEDNESS',
             'RIGHT_INPUT_SHUFFLEDNESS',
             'HASH_TABLES_DISTINCT_VALUE_COUNT',
             'HASH_TABLES_POSITION_COUNT',
             'left_input_is_reference_table',
             'right_input_is_reference_table',
             'execution_time_ms']]

    hash_joins = df.query('RADIX_BITS >= 0.0')
    sm_joins = df.query('RADIX_BITS.isna()', engine='python')


    start = timer()
    hash_joins_materialize = process_hash_join_materialize_rows(hash_joins.copy())
    mid_1 = timer()
    hash_joins_remainder = process_hash_join_remainder_rows(hash_joins.copy())
    mid_2 = timer()
    sm_joins = sm_joins[[
             'QUERY_HASH',
             'OPERATOR_HASH',
             'JOIN_MODE',
             'PRIMARY_PREDICATE',
             'PREDICATE_COUNT',
             'left_input_is_reference_table',
             'DISTINCT_VALUE_COUNT_RIGHT',
             'segment_encoding_spec_left',
             'LEFT_TABLE_ROW_COUNT',
             'RIGHT_TABLE_ROW_COUNT',
             'segment_encoding_spec_right',
             'DISTINCT_VALUE_COUNT',
             'INPUT_SHUFFLEDNESS',
             'right_input_is_reference_table',
             'LEFT_TABLE_CHUNK_COUNT',
             'RIGHT_TABLE_CHUNK_COUNT',
             'execution_time_ms']].copy()
    sm_joins_processed = process_sort_merge_joins(sm_joins)
    end = timer()

    # When only projection the actual feature columns, we expect no NAN values.
    hjm_projection = hash_joins_materialize[encoding_selection_constants.PREDICTION_COLUMNS['hash_join_materialize']]
    if len(hjm_projection[hjm_projection.isnull().any(axis=1)]) > 0:
        print(hjm_projection[hjm_projection.isnull().any(axis=1)])
    assert len(hjm_projection[hjm_projection.isnull().any(axis=1)]) == 0
    hjr_projection = hash_joins_remainder[encoding_selection_constants.PREDICTION_COLUMNS['hash_join_remainder']]
    assert len(hjr_projection[hjr_projection.isnull().any(axis=1)]) == 0
    smj_projection = sm_joins_processed[encoding_selection_constants.PREDICTION_COLUMNS['sort_merge_join']]
    assert len(smj_projection[smj_projection.isnull().any(axis=1)]) == 0

    return (hash_joins_materialize, hash_joins_remainder, sm_joins_processed)


def write_join_data(df_materialize, df_remainder, df_sort_merge, write_folder):
    materialize_file = os.path.join(write_folder, 'hash_join_materialize_prepared.csv.bz2')
    df_materialize.to_csv(materialize_file, sep = ',', index = False)
    remainder_file = os.path.join(write_folder, 'hash_join_remainder_prepared.csv.bz2')
    df_remainder.to_csv(remainder_file, sep = ',', index = False)
    sort_merge_file = os.path.join(write_folder, 'sort_merge_join_prepared.csv.bz2')
    df_sort_merge.to_csv(sort_merge_file, sep = ',', index = False)


#######
#######
#######     PROJECTIONS
#######
#######

def add_features_projections(df):
    # We neglect pretty much all columns that do not require a calculation as they are usually just forwarded.

    columns_to_append = {}

    # We know for every column whether it is forwared or calculated. Before grouping per operator instance, we collect
    # this information and later use  SUM() on the generated columns.
    for data_type in encoding_selection_constants.DATA_TYPES:
        data_type_marker = np.where(data_type == df.DATA_TYPE, 1.0, 0.0)
        for encoding in encoding_selection_constants.ENCODINGS:
            sequential_from_data_table = (1 - df.input_is_reference_table)
            sequential_from_ref_table = df.input_is_reference_table * (1 - df.INPUT_SHUFFLEDNESS)
            nonseq_from_ref_table = df.input_is_reference_table * df.INPUT_SHUFFLEDNESS

            columns_to_append[f'{data_type}_cells_processed_with_{encoding}_from_data_table'] = np.where(df['segment_encoding_spec'] == encoding,
                                                                                          df.REQUIRES_COMPUTATION * data_type_marker * \
                                                                                          sequential_from_data_table * df.INPUT_ROW_COUNT,
                                                                                          0.0)
            columns_to_append[f'{data_type}_cells_processed_with_{encoding}_from_seq_ref_table'] = np.where(df['segment_encoding_spec'] == encoding,
                                                                                             df.REQUIRES_COMPUTATION * data_type_marker * \
                                                                                             sequential_from_ref_table * df.INPUT_ROW_COUNT,
                                                                                             0.0)
            columns_to_append[f'{data_type}_cells_processed_with_{encoding}_nonseq'] = np.where(df['segment_encoding_spec'] == encoding,
                                                                                 df.REQUIRES_COMPUTATION * data_type_marker * \
                                                                                 nonseq_from_ref_table * df.INPUT_ROW_COUNT,
                                                                                 0.0)

    columns_to_append['CASE_WHEN_processed'] = df['DESCRIPTION'].str.count(" CASE WHEN ") * df.INPUT_ROW_COUNT
    columns_to_append['OR_processed'] = df['DESCRIPTION'].str.count(" OR ") * df.INPUT_ROW_COUNT

    comma_count = df['DESCRIPTION'].str.count(",")
    substr_processed = df['DESCRIPTION'].str.count("SUBSTR")
    # substr has two commas we want to ignore, there are not cases of 0 commas but substr functions
    columns_to_append["clause_count"] = (comma_count - (2 * substr_processed) + 1)

    tmp_df = pd.DataFrame(columns_to_append)
    rows_before_concat = len(df)
    assert rows_before_concat == len(tmp_df)
    df = pd.concat([df, tmp_df], axis=1)
    assert len(df) == rows_before_concat, "pd.concat yielded new rows. I don't want that."

    grouped = df.groupby(['QUERY_HASH', 'OPERATOR_HASH'])

    # grouping and using max for "boring" features such as input size which is equal for all rows of an aggregation operator
    grouped_max = grouped[['QUERY_HASH', # used to later obtain the initial query
                           'OPERATOR_HASH',
                           'INPUT_ROW_COUNT',
                           'INPUT_CHUNK_COUNT',
                           'OUTPUT_ROW_COUNT',
                           'OUTPUT_CHUNK_COUNT',
                           'DESCRIPTION',
                           'input_row_count_log',
                           'execution_time_ms',
                           'REQUIRES_COMPUTATION']].max()
    # For some columns, we take a simple mean/sum of all columns' properties.
    agg_dict = {'INPUT_SHUFFLEDNESS': 'mean',
                'input_is_reference_table': 'mean',
                'DISTINCT_VALUE_COUNT': 'mean'}
    for data_type in encoding_selection_constants.DATA_TYPES:
        for encoding in encoding_selection_constants.ENCODINGS:
            agg_dict[f'{data_type}_cells_processed_with_{encoding}_from_data_table'] = 'sum'
            agg_dict[f'{data_type}_cells_processed_with_{encoding}_from_seq_ref_table'] = 'sum'
            agg_dict[f'{data_type}_cells_processed_with_{encoding}_nonseq'] = 'sum'

    agg_dict['CASE_WHEN_processed'] = 'min'
    agg_dict['OR_processed'] = 'min'
    agg_dict['clause_count'] = 'min'
    agg_dict['INPUT_ROW_COUNT'] = 'min'  # needed for wide_projection_feature

    custom_column_aggs = grouped.agg(agg_dict)
    column_counts = grouped.size().reset_index()

    # There is a rather large performance difference between a wide projection that calculate on few columns (e.g.,
    # `a + 1, b + 2, c + 3`) and wide projection with few columns in the output but many columns in the calculation
    # (e.g., a + b * (c - 3)). This feature tries to roughly capture this overhead.
    custom_column_aggs["column_count"] = grouped.size()
    custom_column_aggs["wide_projection_cells"] = np.power(custom_column_aggs["column_count"] / custom_column_aggs["clause_count"], 2.0) * custom_column_aggs.INPUT_ROW_COUNT

    # We multiply later (i.e., here) as we need the clause count in the wide_projection feature.
    # Clauses processed helps to somewhat cover the overhead of a complex expression over multiple columns. We have
    # already incorporated the materializations for such expressions. But the following actual calculation also comes
    # with significant costs.
    custom_column_aggs["clauses_processed"] = custom_column_aggs.clause_count * custom_column_aggs.INPUT_ROW_COUNT
    custom_column_aggs = custom_column_aggs.drop(columns=["INPUT_ROW_COUNT"])  # no longer needed, avoid name clash when appending

    requires_computation_agg = df.groupby(['QUERY_HASH', 'OPERATOR_HASH', 'REQUIRES_COMPUTATION']).REQUIRES_COMPUTATION.sum().unstack()

    data_types_agg = df.groupby(['QUERY_HASH', 'OPERATOR_HASH', 'DATA_TYPE']).DATA_TYPE.count().unstack()
    encodings_agg = df.groupby(['QUERY_HASH', 'OPERATOR_HASH', 'segment_encoding_spec']).segment_encoding_spec.count().unstack()

    result = pd.concat([grouped_max, custom_column_aggs, requires_computation_agg, data_types_agg, encodings_agg], axis=1, sort=False).fillna(0)
    for data_type_to_add in ['float', 'int', 'string']:
        # add in case hasn't occurred in training data
        if data_type_to_add not in result.columns:
            result[data_type_to_add] = 0.0

    result = result.rename({1.0: 'cells_requiring_computation',
                            0.0: 'cells_not_requiring_computation',
                            'float': 'float_cells_to_process',
                            'int': 'integer_cells_to_process',
                            'string': 'string_cells_to_process',
                            'INPUT_SHUFFLEDNESS': 'avg_input_sequential_factor',
                            'input_is_reference_table': 'cells_read_from_reference_table',
                            'DISTINCT_VALUE_COUNT': 'avg_distinct_value_count'}, axis='columns')

    result["chunks_per_row"] = result.INPUT_CHUNK_COUNT / (result.INPUT_ROW_COUNT + 1)

    return result


def get_projection_data(read_folder):
    filepath = os.path.join(read_folder, 'projection.csv.bz2')
    if not Path(filepath).exists():
        return None

    projections = pd.read_csv(filepath, low_memory=False)
    return projections


def featurize_projections(df, process_count = int(multiprocessing.cpu_count() / 2), verbose = False):
    start = timer()
    add_features(df)
    projections = add_features_projections(df)
    end = timer()

    if len(projections[projections.isnull().any(axis=1)]) > 0:
        print(projections[projections.isnull().any(axis=1)])
    assert len(projections[projections.isnull().any(axis=1)]) == 0

    return projections


def write_projection_data(df, write_folder):
    file = os.path.join(write_folder, 'projection_prepared.csv.bz2')
    df.to_csv(file, sep = ',', index = False)


#######
#######
#######     ALL OPERATORS
#######
#######

def featurize_all(read_folder, write_folder, process_count = int(multiprocessing.cpu_count() / 2), verbose = False):
    def table_scans(read_folder, write_folder):
        print(f'{datetime.datetime.now()}: Table scans - start processing.')
        start = timer()
        table_scans = get_table_scan_data(read_folder)
        featurize = timer()
        table_scans_adapted = featurize_table_scans(table_scans, process_count, verbose)
        writing = timer()
        write_table_scan_data(table_scans_adapted, write_folder)
        end = timer()
        print(f'{datetime.datetime.now()}: Table scans - finished (loading: {featurize - start:,.2f} s, featurization: {writing - featurize:,.2f} s, writing: {end - featurize:,.2f} s).')

    def aggregates(read_folder, write_folder):
        print(f'{datetime.datetime.now()}: Aggregates - start processing.')
        start = timer()
        aggregates = get_aggregate_data(read_folder)
        featurize = timer()
        aggregates_adapted = featurize_aggregates(aggregates, process_count, verbose)
        writing = timer()
        write_aggregate_data(aggregates_adapted, write_folder)
        end = timer()
        print(f'{datetime.datetime.now()}: Aggregates - finished (loading: {featurize - start:,.2f} s, featurization: {writing - featurize:,.2f} s, writing: {end - featurize:,.2f} s).')

    def joins(read_folder, write_folder):
        print(f'{datetime.datetime.now()}: Joins - start processing.')
        start = timer()
        joins = get_join_data(read_folder)
        featurize = timer()
        hash_joins_materialize, hash_joins_remainder, sort_merge_joins = featurize_joins(joins, process_count, verbose)
        writing = timer()
        write_join_data(hash_joins_materialize, hash_joins_remainder, sort_merge_joins, write_folder)
        end = timer()
        print(f'{datetime.datetime.now()}: Joins - finished (loading: {featurize - start:,.2f} s, featurization: {writing - featurize:,.2f} s, writing: {end - featurize:,.2f} s).')

    table_scans_process = multiprocessing.Process(target=table_scans, args=(read_folder, write_folder, ))
    table_scans_process.start()

    aggregates_process = multiprocessing.Process(target=aggregates, args=(read_folder, write_folder, ))
    aggregates_process.start()

    joins_process = multiprocessing.Process(target=joins, args=(read_folder, write_folder, ))
    joins_process.start()

    print(f'{datetime.datetime.now()}: Projections - start processing.')
    start = timer()
    projections = get_projection_data(read_folder)
    if projections is not None:
        featurize = timer()
        projections_adapted = featurize_projections(projections, process_count, verbose)
        writing = timer()
        write_projection_data(projections_adapted, write_folder)
        end = timer()
        print(f'{datetime.datetime.now()}: Projections - finished (loading: {featurize - start:,.2f} s, featurization: {writing - featurize:,.2f} s, writing: {end - featurize:,.2f} s).')
    else:
        print(f'{datetime.datetime.now()}: Projections - skipped. No projections in workload.')

    table_scans_process.join()
    aggregates_process.join()
    joins_process.join()

