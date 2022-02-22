#!/usr/bin/env python3

import enum


@enum.unique
class EncodingType(enum.Enum):
    dictionary_fwi = 0
    dictionary_bp = 1
    frameofreference_fwi = 2
    frameofreference_bp = 3
    fixedstringdictionary_fwi = 4
    fixedstringdictionary_bp = 5
    fsst_bp = 6
    unencoded = 7
    runlength = 8
    lz4_bp = 9


SUPPORTED_DATA_TYPES = [['int', 'long', 'float', 'double', 'string'],
                        ['int', 'long', 'float', 'double', 'string'],
                        ['int'],
                        ['int'],
                        ['string'],
                        ['string'],
                        ['string'],
                        ['int', 'long', 'float', 'double', 'string'],
                        ['int', 'long', 'float', 'double', 'string'],
                        ['int', 'long', 'float', 'double', 'string']]

DATA_TYPES = ['int', 'long', 'float', 'double', 'string']

ENCODINGS = [encoding.name for encoding in EncodingType]

JOIN_MODES = ['Inner', 'Left', 'Right', 'FullOuter', 'Cross', 'Semi', 'AntiNullAsTrue', 'AntiNullAsFalse']

PREDICATE_CONDITIONS = ['Equals', 'NotEquals', 'LessThan', 'LessThanEquals', 'GreaterThan', 'GreaterThanEquals',
                        'BetweenInclusive', 'BetweenLowerExclusive', 'BetweenUpperExclusive', 'BetweenExclusive',
                        'In', 'NotIn', 'Like', 'NotLike', 'IsNull', 'IsNotNull']

_prediction_columns_table_scan = [
    'OUTPUT_ROW_COUNT',
    'DISTINCT_VALUE_COUNT',
    'expression_evaluator_in_list_element_count',
    'expression_evaluator_case_count',
    'expression_evaluator_OR_count',
    'execution_time_ms',
    'chunks_per_row']
for data_type in DATA_TYPES:
    for encoding in ENCODINGS:
        for side in ["left", "right"]:
            for predicate_name in ["preceding_like_pattern", "trailing_like_pattern", "double_like_pattern",
                                   "column_vs_column", "expression_evaluator", "subquery", "simple_predicate",
                                   "between"]:
                if side == "right" and predicate_name != "column_vs_column":
                    continue

                if "like" in predicate_name:
                    if data_type != "string":
                        continue
                    # handle "like" and "not like"
                    for like_predicate_name in [predicate_name, predicate_name.replace("_like_", "_notlike_")]:
                        _prediction_columns_table_scan.append(f"{side}_cells_read__{like_predicate_name}__{data_type}_{encoding}_from_data_table")
                        _prediction_columns_table_scan.append(f"{side}_cells_read__{like_predicate_name}__{data_type}_{encoding}_from_seq_ref_table")
                        _prediction_columns_table_scan.append(f"{side}_cells_read__{like_predicate_name}__{data_type}_{encoding}_from_nonseq_ref_table")
                else:
                    # There are no like predicates for column-vs-column scans
                    _prediction_columns_table_scan.append(f"{side}_cells_read__{predicate_name}__{data_type}_{encoding}_from_data_table")
                    _prediction_columns_table_scan.append(f"{side}_cells_read__{predicate_name}__{data_type}_{encoding}_from_seq_ref_table")
                    _prediction_columns_table_scan.append(f"{side}_cells_read__{predicate_name}__{data_type}_{encoding}_from_nonseq_ref_table")

_prediction_columns_aggregate = []
for data_type in DATA_TYPES:
    for encoding in ENCODINGS:
        for name in ["single_column", "two_columns", "non_opt"]:
            _prediction_columns_aggregate.append(f"{data_type}_cells_read_in_{name}_groupby_columns_with_{encoding}_from_data_table")
            _prediction_columns_aggregate.append(f"{data_type}_cells_read_in_{name}_groupby_columns_with_{encoding}_from_seq_ref_table")
            _prediction_columns_aggregate.append(f"{data_type}_cells_read_in_{name}_groupby_columns_with_{encoding}_nonseq")
        for agg_name in ["countstar", "calculation"]:
            _prediction_columns_aggregate.append(f"{data_type}_cells_read_in_{agg_name}_aggregate_columns_with_{encoding}_from_data_table")
            _prediction_columns_aggregate.append(f"{data_type}_cells_read_in_{agg_name}_aggregate_columns_with_{encoding}_from_seq_ref_table")
            _prediction_columns_aggregate.append(f"{data_type}_cells_read_in_{agg_name}_aggregate_columns_with_{encoding}_nonseq")
_prediction_columns_aggregate.extend([
        'non_consecutive_aggregation_calculations',
        'OUTPUT_ROW_COUNT',
        'avg_distinct_values_groupby_columns',
        'avg_distinct_values_aggregate_columns',
        'chunks_per_row',
        'execution_time_ms'])

_prediction_columns_projection = []
for data_type in DATA_TYPES:
        for encoding in ENCODINGS:
            _prediction_columns_projection.append(f'{data_type}_cells_processed_with_{encoding}_from_data_table')
            _prediction_columns_projection.append(f'{data_type}_cells_processed_with_{encoding}_from_seq_ref_table')
            _prediction_columns_projection.append(f'{data_type}_cells_processed_with_{encoding}_nonseq')
_prediction_columns_projection.extend([
        'OUTPUT_ROW_COUNT',
        'chunks_per_row',
        'CASE_WHEN_processed',
        'OR_processed',
        'clauses_processed',
        'wide_projection_cells',
        'execution_time_ms'])

_prediction_columns_hash_join_materialize = []
for data_type in DATA_TYPES:
    for encoding in ENCODINGS:
        _prediction_columns_hash_join_materialize.append(f"cells_read__{data_type}_{encoding}__data_table")
        _prediction_columns_hash_join_materialize.append(f"cells_read__{data_type}_{encoding}__seq_ref_table")
        _prediction_columns_hash_join_materialize.append(f"cells_read__{data_type}_{encoding}__nonseq_ref_table")
        _prediction_columns_hash_join_materialize.append(f"cells_materialized__{data_type}_{encoding}__data_table")
        _prediction_columns_hash_join_materialize.append(f"cells_materialized__{data_type}_{encoding}__seq_ref_table")
        _prediction_columns_hash_join_materialize.append(f"cells_materialized__{data_type}_{encoding}__nonseq_ref_table")
_prediction_columns_hash_join_materialize.extend([
        'estimated_dictionary_searches',
        'execution_time_ms',
        'chunks_per_row'
        ])

_prediction_columns_hash_join_remainder = []
for data_type in DATA_TYPES:
    _prediction_columns_hash_join_remainder.append(f"cells_to_cluster_{data_type}")
    _prediction_columns_hash_join_remainder.append(f"cells_for_semi_anti_build_phase_{data_type}")
    _prediction_columns_hash_join_remainder.append(f"cells_for_non_semi_anti_build_phase_{data_type}")
    _prediction_columns_hash_join_remainder.append(f"rows_to_probe_semi_anti_like_{data_type}")
    _prediction_columns_hash_join_remainder.append(f"rows_to_probe_non_semi_anti_inner_like_{data_type}")
    _prediction_columns_hash_join_remainder.append(f"rows_to_probe_remainer_{data_type}")
_prediction_columns_hash_join_remainder.extend([
        'rows_probed_for_secondary_predicates',
        'execution_time_ms',
        'output_row_count'])

_prediction_columns_sort_merge_join = []
for join_mode in JOIN_MODES:
    _prediction_columns_sort_merge_join.append(f'join_mode_{join_mode}')
for predicate_condition in PREDICATE_CONDITIONS:
    _prediction_columns_sort_merge_join.append(f'predicate_condition_{predicate_condition}')
_prediction_columns_sort_merge_join.extend([
        'left_distinct_value_count',
        'right_distinct_value_count',
        'left_input_row_count_log',
        'right_input_row_count_log',
        'rows_joined_not_inner',
        'rows_joined_one_sided_outer_join',
        'rows_joined_full_outer_join',
        'rows_joined_non_equi',
        'rows_joined_for_secondary_predicates',
        'left_sort_complexity',
        'right_sort_complexity',
        'execution_time_ms'])


PREDICTION_COLUMNS = {'table_scan': _prediction_columns_table_scan,
                      'aggregate': _prediction_columns_aggregate,
                      'projection': _prediction_columns_projection,
                      'hash_join_materialize': _prediction_columns_hash_join_materialize,
                      'hash_join_remainder': _prediction_columns_hash_join_remainder,
                      'sort_merge_join': _prediction_columns_sort_merge_join}

