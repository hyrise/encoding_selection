#!/usr/bin/env python3

import os
import re
import sys  # for sys.maxsize & path
import glob
import json
import math
import time
import joblib  # loading models
import random
import hashlib
import numbers
import datetime
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import expon  # synthetic data generation
from timeit import default_timer as timer
from pandas.testing import assert_frame_equal

sys.path.append("..")
from helpers import encoding_selection_constants
from helpers import encoding_selection_helpers
from helpers import dag_traversal_helpers
from helpers import feature_preparation_helpers
from helpers import size_estimation
from selection import selection_approaches
from selection import column_runtime_change_prediction_helper


def get_distinct_value_count(df, adapt_to_single_chunk_predictions = True):
    df['shady_distinct_count_estimation'] = df['DISTINCT_VALUE_COUNT'] / np.power(df['ROW_COUNT_TAB'] / df['ROW_COUNT'], 0.7)
    df['first_column_distinct_value_count'] = df['DISTINCT_VALUE_COUNT']
    if adapt_to_single_chunk_predictions:
        df['first_column_distinct_value_count'] = df[['ROW_COUNT', 'shady_distinct_count_estimation']].min(axis=1)
        
    return df


# Adds the column required for grouping. It can happen that the result of predicted operatores is either empty or just
# stores operatores that never access data columns (e.g., an aggregate on a temporary table).
def add_columns_for_grouping(df):
    df['TABLE_NAME'] = np.nan
    df['COLUMN_NAME'] = np.nan
    df['ENCODING_TYPE'] = np.nan
    df['VECTOR_COMPRESSION_TYPE'] = np.nan
    df['change'] = 0.0


def add_prediction_features(df, adapt_to_single_chunk_predictions = True):
    tmp_column_names_to_drop = []
    df['effective_chunk_count'] = np.ceil(df['ROW_COUNT_TAB'] / df['MAX_CHUNK_SIZE'])
    if adapt_to_single_chunk_predictions:
        df['effective_chunk_count'] = 1

    df = get_distinct_value_count(df, adapt_to_single_chunk_predictions)
    df['first_column_is_reference_segment'] = (df['SCAN_TYPE'] == 'REFERENCE_SCAN').astype(int)
    
    df['left_input_data_table_row_count'] = df['INPUT_ROWS']
    df['left_input_row_count_log'] = np.log10(df['INPUT_ROWS'])
    if adapt_to_single_chunk_predictions:
        df['left_input_data_table_row_count'] = df['ROW_COUNT']
        df['left_input_row_count_log'] = np.log10(df['ROW_COUNT'])

    df['selectivity'] = df['OUTPUT_ROWS'] / df['INPUT_ROWS']
    df['is_selectivity_below_50_percent'] = (df.selectivity < 0.5).astype(int)
    df['selectivity_distance_to_50_percent'] = (df.selectivity - 0.5).abs()

    df['is_column_comparison'] = df.DESCRIPTION.str.contains('ColumnVsColumn', regex=False).astype(int)
    # LIKE
    df['match_pattern__like'] = df.COLUMN_NAME + " LIKE "
    df['scan_operator_type_LIKE'] = [x[0] in x[1] for x in zip(df.match_pattern__like, df.DESCRIPTION)]
    df['scan_operator_type_LIKE'] = df['scan_operator_type_LIKE'].astype(int)
    tmp_column_names_to_drop.append('match_pattern__like')
    # EQUALS
    df['match_pattern__equals'] = df.COLUMN_NAME + " = "
    df['scan_operator_type_='] = [x[0] in x[1] for x in zip(df.match_pattern__equals, df.DESCRIPTION)]
    df['scan_operator_type_='] = df['scan_operator_type_='].astype(int)
    tmp_column_names_to_drop.append('match_pattern__equals')
    # INEQUALITY
    df['match_pattern__something_less'] = df.COLUMN_NAME + " <"  # hits < and <= 
    df['match_pattern__something_greater'] = df.COLUMN_NAME + " >"  # hits > and >=
    df['scan_operator_type_LESS'] = [x[0] in x[1] for x in zip(df.match_pattern__something_less, df.DESCRIPTION)]
    df['scan_operator_type_GREATER'] = [x[0] in x[1] for x in zip(df.match_pattern__something_greater, df.DESCRIPTION)]
    df['scan_operator_type_LessThanEquals'] = df.scan_operator_type_LESS | df.scan_operator_type_GREATER
    df['scan_operator_type_LessThanEquals'] = df['scan_operator_type_LessThanEquals'].astype(int)
    tmp_column_names_to_drop.extend(['scan_operator_type_LESS', 'scan_operator_type_GREATER',
                                     'match_pattern__something_less', 'match_pattern__something_greater'])

    # BETWEEN
    df['match_pattern__between'] = df.COLUMN_NAME + " BETWEEN "
    df['scan_operator_type_BETWEEN_INCLUSIVE'] = [x[0] in x[1] for x in zip(df.match_pattern__between, df.DESCRIPTION)]
    df['scan_operator_type_BETWEEN_INCLUSIVE'] = df['scan_operator_type_BETWEEN_INCLUSIVE'].astype(int)
    tmp_column_names_to_drop.append('match_pattern__between')
    # REMAINING: we use the OR feature to cover all "expensive" cases which usually involve the expression evaluator
    df['scan_operator_type_Or'] = df.DESCRIPTION.str.contains(' OR ', regex=False).astype(int)
    df['scan_operator_simple_predicate'] = df['scan_operator_type_LIKE'] + df['scan_operator_type_='] + df['scan_operator_type_LessThanEquals'] + df['scan_operator_type_BETWEEN_INCLUSIVE']
    df['scan_operator_type_Or'] = df.scan_operator_type_Or | (df['scan_operator_simple_predicate'] == 0)
    df['scan_operator_type_Or'] = df.scan_operator_type_Or.astype(int)
    tmp_column_names_to_drop.append('scan_operator_simple_predicate')
    # TODO: is_column_column should be distinct from scan_= ... equal scans are cheap with SIMD.
    # For column scans, the predicate is probably irrelevant?!?
    
    df['tmp_distinct_values'] = 2.0
    df['tmp_distinct_values'] = df[['tmp_distinct_values', 'first_column_distinct_value_count']].max(axis=1)
    df['min_segment_av_width'] = 8.0
    df['segment_av_width'] = np.power(2, np.ceil(np.log2(np.log2(df.tmp_distinct_values))))
    df['segment_av_width'] = df[['min_segment_av_width', 'segment_av_width']].max(axis=1)
    df['tmp_min_dictionary_searches'] = 0.0
    df['dictionary_searches'] = (1 - df.first_column_is_reference_segment) * np.log2(df.tmp_distinct_values)
    df['dictionary_searches'] = df[['tmp_min_dictionary_searches', 'dictionary_searches']].max(axis=1)
    tmp_column_names_to_drop.extend(['tmp_distinct_values', 'min_segment_av_width', 'tmp_min_dictionary_searches'])
    
    df['tmp_is_like_column_scan'] = df.DESCRIPTION.str.startswith('TableScan Impl: ColumnLike')
    df['tmp_is_preceding_like'] = df.DESCRIPTION.str.contains(r" LIKE '%.*'", regex = True, case = False)
    df['tmp_is_trailing_like'] = df.DESCRIPTION.str.contains(r" LIKE '[^%]+%'", regex = True, case = False)
    df['string_scan_preceding_like'] = (df.tmp_is_like_column_scan & df.tmp_is_preceding_like).astype(int)
    df['string_scan_trailing_like'] = (df.tmp_is_like_column_scan & df.tmp_is_trailing_like).astype(int)
    tmp_column_names_to_drop.extend(['tmp_is_like_column_scan', 'tmp_is_preceding_like', 'tmp_is_trailing_like'])

    # dropping the columns costs up to 100ms (of 800ms combined) for TPC-H data set, but we still consider it worth
    # it, because it potentially (depending on what Python decides to do with the freed memory) reduces the memory
    # consumption of a data structure we'll hold for quite some time.
    df = df.drop(columns=tmp_column_names_to_drop)

    return df


def parse_file_based_workload(workload_folder, runtime_models_folder, size_models_folder, load_table_scans=True,
                              load_aggregations=True, load_joins=True, load_projections=True, melt_results=True,
                              model="XGBoostReg"):
    split_config = None
    split_config_filename = Path(runtime_models_folder) / ".." / ".." / "split_config.json"
    if split_config_filename.exists():
        with open(split_config_filename) as split_config_file:
            split_config = json.load(split_config_file)
            split_value = float(split_config["split_value"])
            split_run_id = int(split_config["split_run_id"])

    # stores the processed operator data
    processed_for_selection_folder = os.path.join(workload_folder, "..", "processed_for_selection_folder")
    os.makedirs(processed_for_selection_folder, exist_ok = True)

    plan_cache = pd.read_csv(f'{workload_folder}/plan_cache.csv')

    table_meta_data = pd.read_csv(f'{workload_folder}/meta_data_table.csv', index_col = 'TABLE_NAME')
    table_meta_data = table_meta_data.sort_index()
    table_meta_data['TABLE_ID'] = range(0, len(table_meta_data))
    column_meta_data = pd.read_csv(f'{workload_folder}/meta_data_column.csv',
                                      index_col = ['TABLE_NAME', 'COLUMN_NAME'])
    column_meta_data['COLUMN_ID'] = column_meta_data.groupby(['TABLE_NAME']).cumcount()
    segment_meta_data = pd.read_csv(f'{workload_folder}/meta_data_segment.csv', index_col = ['TABLE_NAME', 'COLUMN_NAME', 'CHUNK_ID'])
    chunk_meta_data = pd.read_csv(f'{workload_folder}/meta_data_chunk.csv', index_col = ['TABLE_NAME', 'CHUNK_ID'])
    workload = {}

    # static variables used for three-dimensional configuration matrix
    __table_count = len(table_meta_data)
    __max_row_clusters = segment_meta_data.reset_index()['CHUNK_ID'].max() + 1
    __max_column_count = column_meta_data.reset_index().groupby(['TABLE_NAME'])['TABLE_NAME'].agg(['count'])['count'].max()

    __table_dimensions = []
    __table_names = table_meta_data.reset_index().sort_values(by=['TABLE_ID'])['TABLE_NAME'].tolist()
    for table_name in __table_names: # Order based on increasing TABLE_ID. Could be more explicit.
        row_cluster_count = segment_meta_data.reset_index().query(f"TABLE_NAME == '{table_name}'")['CHUNK_ID'].max() + 1
        column_count = len(column_meta_data.reset_index().query(f"TABLE_NAME == '{table_name}'"))
        __table_dimensions.append([row_cluster_count, column_count])

    ####
    #### SIZE ESTIMATIONS
    ####
    pandas_size_information_file_name = os.path.join(processed_for_selection_folder, 'calculated_segment_sizes.csv.bz2')
    pandas_size_information_melted_file_name = os.path.join(processed_for_selection_folder, 'calculated_segment_sizes_melted.csv.bz2')
    numpy_size_information_file_name = os.path.join(processed_for_selection_folder, 'calculated_segment_sizes.npy')
    skip_size_estimations = False
    if not skip_size_estimations:
        if not os.path.isfile(pandas_size_information_file_name) or not os.path.isfile(numpy_size_information_file_name):
            print(f'{datetime.datetime.now()}: Starting size prediction.')

            # create and load "featurized" segment file
            size_estimation.create_size_model_learning_data(workload_folder, processed_for_selection_folder)
            __segment_sizes = pd.read_csv(os.path.join(processed_for_selection_folder, 'size_model_prepared.csv.bz2'))

            # Join table meta data to add TABLE_ID
            __segment_sizes = pd.merge(__segment_sizes, table_meta_data, how='inner', on=['TABLE_NAME'], suffixes=('', '_TAB'))
            __segment_sizes_projected = __segment_sizes.reset_index()[['DISTINCT_VALUE_COUNT', 'ROW_COUNT', 'row_count_log', 'segment_av_width', 'nullable_rows', 'VALUE_SWITCHES', 'AVG_STRING_LENGTH', 'MAX_STRING_LENGTH']]

            model_found = False
            for filename in glob.iglob(f'{size_models_folder}/**', recursive = True):
                if not os.path.isfile(filename) or '_nan' in filename or 'xgb_reg.joblib' not in filename:
                    continue

                model_found = True
                loaded_model = joblib.load(filename)
                column_name = f'prediction__{os.path.basename(os.path.dirname(filename))}'
                predictions = loaded_model.predict(__segment_sizes_projected)
                __segment_sizes[column_name] = predictions

            assert model_found, f"No size models found in path {size_models_folder}"

            # We are almost done here, but we create another numpy array to allow direct access to segment sizes.
            # When numpy arrays are applicable, they are up to 1000x faster in our case here.
            size_prediction_columns = [col for col in __segment_sizes.columns if col.startswith('prediction__')]
            size_projection_columns = size_prediction_columns + ['TABLE_ID','COLUMN_ID','CHUNK_ID','COLUMN_DATA_TYPE']
            sizes_projected = __segment_sizes.reset_index()[size_projection_columns]
            __segment_sizes_melted = pd.melt(sizes_projected, id_vars=['TABLE_ID','COLUMN_ID','CHUNK_ID','COLUMN_DATA_TYPE'],
                                             value_vars=size_prediction_columns, var_name='tmp_ENCODING_TYPE',
                                             value_name='SIZE_IN_BYTES')

            __segment_sizes_melted['DATA_TYPE_MATCHES'] = [x[0].endswith(x[1])
                                                           for x in zip(__segment_sizes_melted['tmp_ENCODING_TYPE'],
                                                                        __segment_sizes_melted['COLUMN_DATA_TYPE'])]
            __segment_sizes_melted = __segment_sizes_melted[__segment_sizes_melted['DATA_TYPE_MATCHES']]
            # not really nice, but ok: from `prediction__runlength_float` and `prediction__dictionary_bp_float` remove the prediction__ and data type
            __segment_sizes_melted['ENCODING_TYPE_string'] = __segment_sizes_melted['tmp_ENCODING_TYPE'].map(lambda x: x[x.find('__')+2:x.rfind('_')])
            __segment_sizes_melted['ENCODING_TYPE'] = __segment_sizes_melted['ENCODING_TYPE_string'].map(lambda x: encoding_selection_constants.EncodingType[x].value)

            __segment_sizes_numpy = np.full((__table_count, __max_row_clusters,
                                             __max_column_count, selection_approaches.COUNT_CONF_OPTIONS), np.finfo(np.float64).max)
            for row in __segment_sizes_melted.itertuples(index=False):
                __segment_sizes_numpy[row.TABLE_ID, row.CHUNK_ID, row.COLUMN_ID, row.ENCODING_TYPE] = row.SIZE_IN_BYTES

            __segment_sizes.to_csv(pandas_size_information_file_name)
            __segment_sizes_melted.to_csv(pandas_size_information_melted_file_name)
            np.save(numpy_size_information_file_name, __segment_sizes_numpy)
        else:
            print(f'{datetime.datetime.now()}: Reading size predictions from disk.')
            __segment_sizes = pd.read_csv(pandas_size_information_file_name)
            __segment_sizes_melted = pd.read_csv(pandas_size_information_melted_file_name)
            __segment_sizes_numpy = np.load(numpy_size_information_file_name)

        __minimal_size = 0
        data_type_prediction_columns = {}
        min_size_predictions = __segment_sizes.copy()
        for data_type in __segment_sizes.COLUMN_DATA_TYPE.unique():
            columns = [col for col in __segment_sizes.columns if col.startswith('prediction__')
                                                       and col.endswith(f'_{data_type}')]
            data_type_prediction_columns[data_type] = columns
            min_size_predictions[f'min_prediction_for_{data_type}'] = __segment_sizes[columns].min(axis=1)

        for data_type in ['int', 'string', 'float']:  # JOB does not contain floats
            column_name = f'min_prediction_for_{data_type}'
            if column_name not in min_size_predictions.columns:
                min_size_predictions[column_name] = 0.0

        min_size_predictions['actual'] = np.where(min_size_predictions.COLUMN_DATA_TYPE == 'int', min_size_predictions.min_prediction_for_int,
                                                  (np.where(min_size_predictions.COLUMN_DATA_TYPE == 'string', min_size_predictions.min_prediction_for_string,
                                                                                                               min_size_predictions.min_prediction_for_float)))
        __minimal_size = min_size_predictions.actual.sum()
        print(f'{datetime.datetime.now()}: Minimal possible size is {__minimal_size/1000/1000:,.2f} MB.')
    else:
        print('WARNING: skipping size estimations.')

    ####
    #### WORKLOAD PARSING:
    ####          This step collects all operator data from a given workload folder and parses the workload to
    ####          (i) obtain DAG information and (ii) create learning data for prediction.
    ####
    current_timestamp = float(datetime.datetime.utcnow().timestamp())
    if not os.path.isfile(os.path.join(processed_for_selection_folder, 'table_scan_prepared.csv.bz2')) or \
            os.path.getmtime(os.path.join(processed_for_selection_folder, 'table_scan_prepared.csv.bz2')) < (current_timestamp - 3600 * 10):
        print(f'{datetime.datetime.now()}: Loading and parsing workload.')
        all_operators = dag_traversal_helpers.load_and_process_calibration_files(workload_folder)
        dag_traversal_helpers.write_results_to_disk(all_operators, processed_for_selection_folder)

        print(f'{datetime.datetime.now()}: Creating predictable data.')
        feature_preparation_helpers.featurize_all(processed_for_selection_folder, processed_for_selection_folder)
    else:
        print(f'{datetime.datetime.now()}: Skipping workload data parsing, files already exist.')


    __runtimes = {}
    runtime_json_path = os.path.join(processed_for_selection_folder, 'predicted_operator_default_runtimes.json')
    Path(runtime_json_path).touch()


    ####
    #### TABLE SCANS
    ####
    if (load_table_scans and not os.path.isfile(os.path.join(processed_for_selection_folder, 'table_scan_runtime_changes.csv.bz2'))):
        print(f'{datetime.datetime.now()}: Predicting scans.')

        __table_scans = column_runtime_change_prediction_helper.collect_unified_operator_runtimes_per_column('table_scan', processed_for_selection_folder, runtime_models_folder, model, True)
        print(f'{datetime.datetime.now()}: Writing table scan changes to disk.')
        __table_scans.to_csv(os.path.join(processed_for_selection_folder, 'table_scan_runtime_changes.csv.bz2'), index=False)
        print(f'{datetime.datetime.now()}: Writing done.')

    elif (load_table_scans):
        print(f'{datetime.datetime.now()}: Loadings scans from file.')
        __table_scans = pd.read_csv(os.path.join(processed_for_selection_folder, 'table_scan_runtime_changes.csv.bz2'), float_precision='high')
        workload['table_scans'] = __table_scans
    else:
        print('Skipping table scans.')

    assert len(__table_scans.query('ENCODING_TYPE == "Dictionary" and VECTOR_COMPRESSION_TYPE == "FixedSize2ByteAligned" and abs(change) > 1e-9')) == 0


    ####
    #### AGGREGATIONS
    ####
    if (load_aggregations and not os.path.isfile(os.path.join(processed_for_selection_folder, 'aggregate_runtime_changes.csv.bz2'))):
        print(f'{datetime.datetime.now()}: Predicting aggregates.')

        __aggregates = column_runtime_change_prediction_helper.collect_unified_operator_runtimes_per_column('aggregate', processed_for_selection_folder, runtime_models_folder, model, True)
        print(f'{datetime.datetime.now()}: Writing aggregate changes to disk.')
        __aggregates.to_csv(os.path.join(processed_for_selection_folder, 'aggregate_runtime_changes.csv.bz2'), index=False)
        print(f'{datetime.datetime.now()}: Writing done.')
    elif (load_aggregations):
        print(f'{datetime.datetime.now()}: Loadings aggregates from file.')
        __aggregates = pd.read_csv(os.path.join(processed_for_selection_folder, 'aggregate_runtime_changes.csv.bz2'), float_precision='high')
        workload['aggregates'] = __aggregates
    else:
        print('Skipping aggregates.')


    ####
    #### JOINS
    ####
    if (load_joins and not os.path.isfile(os.path.join(processed_for_selection_folder, 'join_runtime_changes.csv.bz2'))):
        print(f'{datetime.datetime.now()}: Predicting joins.')

        __joins = column_runtime_change_prediction_helper.collect_join_runtimes_per_column(processed_for_selection_folder, runtime_models_folder, model, True)
        print(f'{datetime.datetime.now()}: Writing join changes to disk.')
        __joins.to_csv(os.path.join(processed_for_selection_folder, 'join_runtime_changes.csv.bz2'), index=False)
        print(f'{datetime.datetime.now()}: Writing done.')
    elif (load_joins):
        print(f'{datetime.datetime.now()}: Loadings joins from file.')
        __joins = pd.read_csv(os.path.join(processed_for_selection_folder, 'join_runtime_changes.csv.bz2'), float_precision='high')
        workload['joins'] = __joins
    else:
        print('Skipping joins.')

    ####
    #### Projections
    ####
    if (load_projections and not os.path.isfile(os.path.join(processed_for_selection_folder, 'projection_runtime_changes.csv.bz2'))):
        print(f'{datetime.datetime.now()}: Predicting projections.')

        __projections = column_runtime_change_prediction_helper.collect_unified_operator_runtimes_per_column('projection', processed_for_selection_folder, runtime_models_folder, model, True)
        if __projections is not None:
            print(f'{datetime.datetime.now()}: Writing projection changes to disk.')
            __projections.to_csv(os.path.join(processed_for_selection_folder, 'projection_runtime_changes.csv.bz2'), index=False)
            print(f'{datetime.datetime.now()}: Writing done.')
        else:
            print(f'{datetime.datetime.now()}: No projections in workload.')
    elif (load_projections):
        print(f'{datetime.datetime.now()}: Loadings projections from file.')
        __projections = pd.read_csv(os.path.join(processed_for_selection_folder, 'projection_runtime_changes.csv.bz2'), float_precision='high')
        workload['projections'] = __projections
    else:
        print('Skipping projections.')



    workload['table_meta_data'] = table_meta_data
    # join to have add the TABLE_ID (reset_index() is required, otherwise index is fully dropped)
    workload['column_meta_data'] = pd.merge(column_meta_data, table_meta_data, how='inner',
                                               left_index=True, right_index=True,
                                               sort=False).reset_index().set_index(['TABLE_ID', 'COLUMN_ID'])
    workload['segment_meta_data'] = segment_meta_data
    
    table_and_column_meta_data = workload['column_meta_data'].reset_index()

    ####
    #### MERGE, MERGE, MERGE ...
    ####
    if melt_results and (load_table_scans or load_aggregations or load_joins or load_projections):
        print(f'{datetime.datetime.now()}: Starting merging phase.')

        results_appended = pd.DataFrame()
        results_grouped = pd.DataFrame()
        if load_table_scans:
            results_appended = pd.concat([results_appended, __table_scans])
            assert __table_scans.groupby(['QUERY_HASH', 'OPERATOR_HASH'], dropna=False).prediction.min().sum() == \
                   __table_scans.groupby(['QUERY_HASH', 'OPERATOR_HASH'], dropna=False).prediction.max().sum()
            table_scans_grouped = __table_scans.groupby(['TABLE_NAME', 'COLUMN_NAME', 'ENCODING_TYPE', 'VECTOR_COMPRESSION_TYPE'],
                                                        dropna=False).agg({'change': 'sum'}).reset_index()
            results_grouped = pd.concat([results_grouped, table_scans_grouped])
        if load_aggregations:
            if 'TABLE_NAME' not in __aggregates.columns:
                add_columns_for_grouping(__aggregates)

            results_appended = pd.concat([results_appended, __aggregates])
            assert __aggregates.groupby(['QUERY_HASH', 'OPERATOR_HASH'], dropna=False).prediction.min().sum() == \
                   __aggregates.groupby(['QUERY_HASH', 'OPERATOR_HASH'], dropna=False).prediction.max().sum()
            aggregates_grouped = __aggregates.groupby(['TABLE_NAME', 'COLUMN_NAME', 'ENCODING_TYPE', 'VECTOR_COMPRESSION_TYPE'],
                                                      dropna=False).agg({'change': 'sum'}).reset_index()
            results_grouped = pd.concat([results_grouped, aggregates_grouped])
        if load_joins:
            results_appended = pd.concat([results_appended, __joins])

            # check that we have only a single prediction value per operator
            assert __joins.groupby(['QUERY_HASH', 'OPERATOR_HASH', 'JOIN_MODEL_TYPE', 'materialize_side']).prediction.min().sum() == \
                   __joins.groupby(['QUERY_HASH', 'OPERATOR_HASH', 'JOIN_MODEL_TYPE', 'materialize_side']).prediction.max().sum()

            # we could also group by join_model_type here, but that shouldn't matter here any longer as we are not
            # interested in the differences between stages
            joins_grouped = __joins.groupby(['TABLE_NAME', 'COLUMN_NAME', 'ENCODING_TYPE', 'VECTOR_COMPRESSION_TYPE']).agg({'change': 'sum'}).reset_index()
            results_grouped = pd.concat([results_grouped, joins_grouped])
        if load_projections and __projections is not None:
            if 'TABLE_NAME' not in __projections.columns:
                add_columns_for_grouping(__projections)

            results_appended = pd.concat([results_appended, __projections])
            # check that we have only a single prediction value per operator
            assert __projections.groupby(['QUERY_HASH', 'OPERATOR_HASH']).prediction.min().sum() == \
                   __projections.groupby(['QUERY_HASH', 'OPERATOR_HASH']).prediction.max().sum()

            projections_grouped = __projections.groupby(['TABLE_NAME', 'COLUMN_NAME', 'ENCODING_TYPE', 'VECTOR_COMPRESSION_TYPE'],
                                                        dropna=False).agg({'change': 'sum'}).reset_index()
            results_grouped = pd.concat([results_grouped, projections_grouped])

            assert len(__projections.query('ENCODING_TYPE == "Dictionary" and VECTOR_COMPRESSION_TYPE == "FixedSize2ByteAligned" and abs(change) > 0.0')) == 0

        # Each join operator might have multiple outputs (materialize and co. for hash joins), we need to get the min
        # values of all operator components (thus include join prediction type in the groupby and disable discarding of
        # nan values).
        predicted_overall_runtime_dict_fwi = results_appended.query('(ENCODING_TYPE == "Dictionary" and VECTOR_COMPRESSION_TYPE == "FixedWidthInteger2Byte") or ' \
                                                                     '(ENCODING_TYPE.isna() and VECTOR_COMPRESSION_TYPE.isna())', engine="python").groupby(['QUERY_HASH',
                                                                                                                                                            'OPERATOR_HASH',
                                                                                                                                                            'JOIN_MODEL_TYPE',
                                                                                                                                                            'materialize_side'], dropna=False).prediction.min().sum()
        print(f'Predicted overall workload runtime with DictionaryFWI for every segment is: {predicted_overall_runtime_dict_fwi:,} ms.')

        assert "dictionary_fwi" in workload_folder, "Expecting a Dictionary (FWI) workload"
        if predicted_overall_runtime_dict_fwi <= 0:
            print((f"WARNING: unexpected predicted dictionary (FWI) runtime ({predicted_overall_runtime_dict_fwi:,} ms)."
                    "If current model is not built on small training set or in development, this result probably signals an error."))

        plan_cache_query_runtimes = plan_cache.EXECUTION_COUNT * plan_cache.AVG_RUNTIME_MS
        overall_runtime_dict_fwi = plan_cache_query_runtimes.sum()
        print(f'Actual overall workload runtime with DictionaryFWI for every segment is: {overall_runtime_dict_fwi:,} ms.')

        ##
        ##  Grouping and neglecting the Query Hash
        ##
        results_grouped = results_appended.groupby(['TABLE_NAME', 'COLUMN_NAME', 'ENCODING_TYPE', 'VECTOR_COMPRESSION_TYPE'], dropna=False).agg({'change': 'sum'}).reset_index()

        results_meta_joined = pd.merge(results_grouped, table_and_column_meta_data, how='left', sort=False, on=['TABLE_NAME', 'COLUMN_NAME'])
        assert len(results_meta_joined) == len(results_grouped)

        encoding_type_to_id = {"segment_encoding_spec": [], "encoding_type_id": []}
        for encoding in encoding_selection_constants.EncodingType:
            encoding_type_to_id["segment_encoding_spec"].append(encoding.name)
            encoding_type_to_id["encoding_type_id"].append(encoding.value)
        encoding_type_to_id_df = pd.DataFrame.from_dict(encoding_type_to_id)
        assert len(encoding_type_to_id_df) == len(encoding_selection_constants.EncodingType)

        results_meta_joined = encoding_selection_helpers.add_segment_encoding_spec_column(results_meta_joined)
        results_meta_encoding_joined = pd.merge(results_meta_joined, encoding_type_to_id_df, how='inner', on='segment_encoding_spec')
        assert len(results_meta_joined) == len(results_meta_encoding_joined)

        print(f'{datetime.datetime.now()}: Creating merged numpy matrixes.')

        greedy_estimated_runtimes = results_appended.copy()
        greedy_estimated_runtimes['tab_col_id'] = greedy_estimated_runtimes.TABLE_NAME + '#' + greedy_estimated_runtimes.COLUMN_NAME
        greedy_estimated_runtimes['col_count'] = greedy_estimated_runtimes.groupby(['QUERY_HASH', 'OPERATOR_HASH', 'JOIN_MODEL_TYPE'], dropna=False).tab_col_id.transform('nunique')
        greedy_estimated_runtimes['runtime_share'] = greedy_estimated_runtimes.prediction / greedy_estimated_runtimes.col_count
        greedy_estimated_runtimes_grouped = greedy_estimated_runtimes.groupby(['TABLE_NAME', 'COLUMN_NAME']).agg({'runtime_share': 'sum'})

        runtimes_list = []
        __runtimes_numpy_matrix = np.full((__table_count, __max_row_clusters, __max_column_count, selection_approaches.COUNT_CONF_OPTIONS), np.nan)
        __runtimes_numpy_matrix_greedy = np.full((__table_count, __max_row_clusters, __max_column_count, selection_approaches.COUNT_CONF_OPTIONS), np.nan)

        print(f'Processing {len(results_meta_encoding_joined)} tuples ', end='')
        for row in results_meta_encoding_joined.itertuples(index=False):
            print('.', end='', flush=True)
            if pd.isna(row.TABLE_ID):
                # operations on temporary columns
                continue

            table_row_count = row.ROW_COUNT
            base_performance = 0.0
            if (row.TABLE_NAME, row.COLUMN_NAME) in greedy_estimated_runtimes_grouped.index:
                base_performance = greedy_estimated_runtimes_grouped.loc[(row.TABLE_NAME, row.COLUMN_NAME), 'runtime_share']

            for chunk_id in range(int(row.CHUNK_COUNT)):
                
                chunk_row_count = chunk_meta_data.loc[(row.TABLE_NAME, chunk_id), 'ROW_COUNT']
                chunk_fraction = float(chunk_row_count) / table_row_count
                runtime_change_chunk = chunk_fraction * row.change
                est_runtime_chunk = chunk_fraction * base_performance
                runtimes_list.append([row.TABLE_ID, chunk_id, row.COLUMN_ID, row.encoding_type_id, runtime_change_chunk])
                __runtimes_numpy_matrix[int(row.TABLE_ID), chunk_id, int(row.COLUMN_ID), int(row.encoding_type_id)] = runtime_change_chunk
                __runtimes_numpy_matrix_greedy[int(row.TABLE_ID), chunk_id, int(row.COLUMN_ID), int(row.encoding_type_id)] = (est_runtime_chunk + runtime_change_chunk) * 1000 * 1000
        print('')

        ##
        ##  Grouping and considering the Query Hash
        ##      We could use these data structures (grouped by query) for everything, but it's measurably slower to
        ##      construct the LP problem with such a large array.
        plan_cache_projection = plan_cache[['QUERY_HASH', 'EXECUTION_COUNT', 'ITEM_NAME']]

        results_appended = pd.merge(results_appended, plan_cache_projection, how='inner', on='QUERY_HASH')

        results_query_grouped = results_appended.groupby(['ITEM_NAME', 'TABLE_NAME', 'COLUMN_NAME',
                                                          'ENCODING_TYPE', 'VECTOR_COMPRESSION_TYPE'], dropna=False).agg({'change': 'sum'}).reset_index()

        results_dict_fwi_grouped = results_appended.groupby(['ITEM_NAME', 'QUERY_HASH', 'OPERATOR_HASH', 'JOIN_MODEL_TYPE', 'materialize_side'], dropna=False).agg({'prediction': 'min'}).reset_index()
        results_dict_fwi_grouped = results_dict_fwi_grouped.groupby(['ITEM_NAME']).agg({'prediction': 'sum'}).reset_index()

        min_query_runtimes = results_appended.groupby(['ITEM_NAME', 'QUERY_HASH', 'OPERATOR_HASH', 'JOIN_MODEL_TYPE', 'materialize_side'], dropna=False).agg({'adapted_prediction': 'min'}).reset_index()
        min_query_runtimes = min_query_runtimes.groupby(['ITEM_NAME']).agg({'adapted_prediction': 'sum'}).reset_index()

        assert (max(predicted_overall_runtime_dict_fwi, results_dict_fwi_grouped['prediction'].sum()) / min(predicted_overall_runtime_dict_fwi, results_dict_fwi_grouped['prediction'].sum())) < 1.01

        plan_cache_executions = plan_cache.groupby("ITEM_NAME").agg({"EXECUTION_COUNT": "sum", "AVG_RUNTIME_MS": "mean"})
        plan_cache_executions["ESTIMATED_SCALE_FACTOR"] = 1
        if "lineitem" in table_meta_data.index:
            plan_cache_executions["ESTIMATED_SCALE_FACTOR"] = int(round(table_meta_data.loc["lineitem"].ROW_COUNT / 6_000_000, 0))
        results_dict_fwi_grouped_with_executions = pd.merge(results_dict_fwi_grouped, plan_cache_executions, how="inner", sort="False", on="ITEM_NAME")
        csv_name = "dict_fwi_runtimes"
        if split_config is not None:
            csv_name += f'__split_{str(split_value).replace(".", "_")}_r{split_run_id}'
        csv_name += f"__{model}.csv"
        results_dict_fwi_grouped_with_executions.to_csv(os.path.join(workload_folder, csv_name), index=False)

        runtimes_dict_fwi_queries = {}
        for row in results_dict_fwi_grouped.itertuples(index=False):
            runtimes_dict_fwi_queries[row.ITEM_NAME] = row.prediction
        
        workload['runtimes_dict_fwi_queries'] = runtimes_dict_fwi_queries

        results_meta_joined = pd.merge(results_query_grouped, table_and_column_meta_data, how='left', sort=False, on=['TABLE_NAME', 'COLUMN_NAME'])
        assert len(results_meta_joined) == len(results_query_grouped)

        encoding_type_to_id = {"segment_encoding_spec": [], "encoding_type_id": []}
        for encoding in encoding_selection_constants.EncodingType:
            encoding_type_to_id["segment_encoding_spec"].append(encoding.name)
            encoding_type_to_id["encoding_type_id"].append(encoding.value)
        encoding_type_to_id_df = pd.DataFrame.from_dict(encoding_type_to_id)
        assert len(encoding_type_to_id_df) == len(encoding_selection_constants.EncodingType)

        results_meta_joined = encoding_selection_helpers.add_segment_encoding_spec_column(results_meta_joined)
        results_meta_encoding_joined = pd.merge(results_meta_joined, encoding_type_to_id_df, how='inner', on='segment_encoding_spec')
        assert len(results_meta_joined) == len(results_meta_encoding_joined)

        print(f'{datetime.datetime.now()}: Creating merged query-containing numpy matrixes.')

        results_meta_encoding_joined_chunks = results_meta_encoding_joined.merge(chunk_meta_data.reset_index(), how="inner", on="TABLE_NAME", suffixes=("_CHANGES", "_CHUNKS"))
        results_meta_encoding_joined_chunks["change_adapted"] = results_meta_encoding_joined_chunks.change * (results_meta_encoding_joined_chunks.ROW_COUNT_CHUNKS / results_meta_encoding_joined_chunks.ROW_COUNT_CHANGES)
        results_meta_encoding_joined_chunks_grouped = results_meta_encoding_joined_chunks.groupby(["ITEM_NAME", "TABLE_ID",
                                                                                                   "COLUMN_ID", "CHUNK_ID",
                                                                                                   "encoding_type_id"]).agg({"change_adapted": "sum"}).reset_index()
        
        runtimes_list_per_query_numpy2 = {}
        for item_name in pd.unique(results_meta_encoding_joined_chunks_grouped.ITEM_NAME):
            filtered = results_meta_encoding_joined_chunks_grouped.query("ITEM_NAME == @item_name")
            runtimes_list_per_query_numpy2[item_name] = filtered[["TABLE_ID", "CHUNK_ID", "COLUMN_ID",
                                                                  "encoding_type_id", "change_adapted"]].to_numpy()

        workload['runtimes_numpy_query'] = runtimes_list_per_query_numpy2

        workload['runtimes_numpy'] = np.asarray(runtimes_list)
        workload['runtimes_numpy_matrix'] = __runtimes_numpy_matrix
        workload['runtimes_numpy_matrix_greedy'] = __runtimes_numpy_matrix_greedy

    workload['segment_sizes'] = __segment_sizes
    workload['segment_sizes_np'] = __segment_sizes_numpy
    workload['table_count'] = __table_count
    workload['table_names'] = __table_names
    workload['max_row_clusters'] = __max_row_clusters
    workload['max_column_count'] = __max_column_count
    workload['table_dimensions'] = __table_dimensions
    workload['minimal_configuration_size'] = __minimal_size

    
    ###
    ### Dictionary-as-the-Default evaluation
    ###
    encoding_selection_helpers.add_segment_encoding_spec_column(__segment_sizes)
    workload['all_dictionary_size'] = __segment_sizes.query("segment_encoding_spec == 'dictionary_fwi'")['SIZE_IN_BYTES'].sum()
    workload['all_dictionary_runtime'] = overall_runtime_dict_fwi

    
    ###
    ### SUPPORTED DATA TYPES
    ###
    # rather inefficient, but data should be of neglectable size
    unsupported_data_types = [] # stores for table_id,column_id an encoding type that is not supported
    for row in table_and_column_meta_data.itertuples(index=False):
        for encoding_id in range(selection_approaches.COUNT_CONF_OPTIONS):
            if row.DATA_TYPE not in encoding_selection_constants.SUPPORTED_DATA_TYPES[encoding_id]:
                unsupported_data_types.append([row.TABLE_ID, row.COLUMN_ID, encoding_id])

    # used by heuristics
    workload['table_and_column_meta_data'] = table_and_column_meta_data.set_index(['TABLE_ID', 'COLUMN_ID'])

    workload['unsupported_data_types_numpy'] = np.array(unsupported_data_types)
    workload['name'] = workload_folder

    print(f'{datetime.datetime.now()}: Done.')

    return workload

