#!/usr/bin/env python3

import collections
import glob
import joblib
import multiprocessing
import numpy as np
import os
import pandas as pd
import sys

from pathlib import Path

sys.path.append("..")
from helpers import feature_preparation_helpers
from helpers import encoding_selection_helpers
from helpers import encoding_selection_constants


ENCODING_TUPLES = [('Dictionary', 'BitPacking'), ('LZ4', 'BitPacking'), ('RunLength', np.nan),
                   ('Unencoded', np.nan), ('FrameOfReference', 'BitPacking'), ('FixedStringDictionary', 'BitPacking'),
                   ('FrameOfReference', 'FixedWidthInteger2Byte'), ('Dictionary', 'FixedWidthInteger2Byte'),
                   ('FixedStringDictionary', 'FixedWidthInteger2Byte'), ('FSST', 'BitPacking')]


def get_initial_operator_data(operator_name, operator_data_folder):
    filepath = os.path.join(operator_data_folder, f'{operator_name}.csv.bz2')
    if not Path(filepath).exists():
        return None

    observations = pd.read_csv(filepath, low_memory=False)
    return observations


#####
#####
#####       TABLE SCAN & AGGREGATE & PROJECTION
#####
#####

# Determines for every query (query hash) the runtime for the baseline encoding (i.e., DictionaryFWI).
# Returns a dataframe that stores the estimated query runtime for the baseline encoding
def predict_baseline(operator_name, df, runtime_models_folder, model = 'heteroscedastic', verbose = False):
    if operator_name == 'table_scan':
        predictable = feature_preparation_helpers.featurize_table_scans(df)
    elif operator_name == 'aggregate':
        predictable = feature_preparation_helpers.featurize_aggregates(df)
    elif operator_name == 'projection':
        predictable = feature_preparation_helpers.featurize_projections(df)
    else:
        exit("Unknown operator.")

    # Right now, we assue the base line is the given workload in form of DictFWI encoding for all columns. This means the passed initial
    # workload is contains mostly dictionary encoded columns and unencoded columns for temporary columns.
    if operator_name == 'aggregate':
        assert predictable.int_cells_read_in_single_column_groupby_columns_with_dictionary_bp_nonseq.max() == 0.0
        assert predictable.int_cells_read_in_two_columns_groupby_columns_with_dictionary_bp_nonseq.max() == 0.0
        assert predictable.string_cells_read_in_two_columns_groupby_columns_with_dictionary_bp_nonseq.max() == 0.0
        assert predictable.string_cells_read_in_two_columns_groupby_columns_with_dictionary_bp_nonseq.max() == 0.0
        assert predictable.int_cells_read_in_single_column_groupby_columns_with_lz4_bp_nonseq.max() == 0.0
        assert predictable.int_cells_read_in_two_columns_groupby_columns_with_lz4_bp_nonseq.max() == 0.0
        assert predictable.string_cells_read_in_two_columns_groupby_columns_with_lz4_bp_nonseq.max() == 0.0
        assert predictable.string_cells_read_in_two_columns_groupby_columns_with_lz4_bp_nonseq.max() == 0.0

    loaded_model = joblib.load(os.path.join(runtime_models_folder, operator_name, f'{model}.joblib'))
    df_projected = predictable[encoding_selection_constants.PREDICTION_COLUMNS[operator_name]]
    df_projected = df_projected.drop(['execution_time_ms'], axis=1)

    predictions = loaded_model.predict(df_projected)
    predictions = encoding_selection_helpers.adapt_predictions(model, predictions)

    # Map all the runtimes back to the columns.
    predictable['prediction'] = predictions
    encoding_selection_helpers.adapt_negative_predictions(predictable, "prediction")

    baseline_execution_time = sum(predictable['prediction'])
    actual_execution_time = sum(predictable['execution_time_ms'])

    # Just to recognize when something went pretty wrong...
    if model in ['heteroscedastic', 'xgb']:
        if len(predictions) > 1000:
            assert actual_execution_time < 5.0 * baseline_execution_time
            assert 5.0 * actual_execution_time > baseline_execution_time

    predictable["prediction_error"] = predictable.prediction - predictable.execution_time_ms

    if verbose:
        print(f'{operator_name}::Error: {(actual_execution_time - baseline_execution_time) / actual_execution_time:.2%} (predicted for '
                      f'baseline: {baseline_execution_time:,.2f}, actual: {actual_execution_time:,.2f})')

    return predictable.reset_index(drop=True)[['QUERY_HASH', 'OPERATOR_HASH', 'prediction']]


def collect_unified_operator_runtimes_per_column(operator_name, operator_data_folder, runtime_models_folder,
                                                 model = 'heteroscedastic', verbose = False):
    df = get_initial_operator_data(operator_name, operator_data_folder)
    if df is None:
        return None

    baseline_execution_times = predict_baseline(operator_name, df, runtime_models_folder, model, verbose)
    loaded_model = joblib.load(os.path.join(runtime_models_folder, operator_name, f'{model}.joblib'))

    if operator_name == "table_scan":
        candidates_left = df[["LEFT_TABLE_NAME", "LEFT_COLUMN_NAME", "DATA_TYPE"]]
        candidates_left.columns = ["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"]
        candidates_right = df[["RIGHT_TABLE_NAME", "RIGHT_COLUMN_NAME", "DATA_TYPE_RIGHT"]]
        candidates_right.columns = ["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"]
        candidates = candidates_left.append(candidates_right)
    else:
        candidates = df[['TABLE_NAME', 'COLUMN_NAME', 'DATA_TYPE']]

    encoding_change_predictions = pd.DataFrame()
    print(f"Processing {len(candidates.drop_duplicates())} change candidates ", end='')
    for row in candidates.drop_duplicates().itertuples(index=False):
        print('.', end='', flush=True)
        table_name = row.TABLE_NAME
        column_name = row.COLUMN_NAME
        data_type = row.DATA_TYPE
        for encoding, vector_compression in ENCODING_TUPLES:

            segment_encoding_spec_df = pd.DataFrame({'ENCODING_TYPE': [encoding], 'VECTOR_COMPRESSION_TYPE': [vector_compression]})
            segment_encoding_spec_str = encoding_selection_helpers.add_segment_encoding_spec_column(segment_encoding_spec_df).loc[0]['segment_encoding_spec']
            if data_type not in encoding_selection_constants.SUPPORTED_DATA_TYPES[int(encoding_selection_constants.EncodingType[segment_encoding_spec_str].value)]:
                continue

            # We do set all rows to the current encoding to predict as an operator might include temporary
            # unencoded columns as well.
            if operator_name == "table_scan":
                df_adapted = df.query('(LEFT_TABLE_NAME == @table_name and LEFT_COLUMN_NAME == @column_name) or \
                                      (RIGHT_TABLE_NAME == @table_name and RIGHT_COLUMN_NAME == @column_name)').copy() 

                for side, suffix in [("LEFT", ""), ("RIGHT", "_RIGHT")]:
                    df_adapted[f'ENCODING_TYPE{suffix}'] = np.where((df_adapted[f'{side}_TABLE_NAME'] == table_name) & (df_adapted[f'{side}_COLUMN_NAME'] == column_name),
                                                           encoding, df_adapted[f'ENCODING_TYPE{suffix}'])
                    df_adapted[f'VECTOR_COMPRESSION_TYPE{suffix}'] = np.where((df_adapted[f'{side}_TABLE_NAME'] == table_name) & (df_adapted[f'{side}_COLUMN_NAME'] == column_name),
                                                                     vector_compression, df_adapted[f'VECTOR_COMPRESSION_TYPE{suffix}'])
            else:
                # We've seen hash collisions before. So we filter on both hashes (not pairs, because I don't know how
                # to do that efficiently) and filter.
                relevant_query_hashes = list(df.query('TABLE_NAME == @table_name and COLUMN_NAME == @column_name').QUERY_HASH.unique())
                relevant_operator_hashes = list(df.query('TABLE_NAME == @table_name and COLUMN_NAME == @column_name').OPERATOR_HASH.unique())
                df_adapted = df[df.QUERY_HASH.isin(relevant_query_hashes) & df.OPERATOR_HASH.isin(relevant_operator_hashes)].copy()
                df_adapted['ENCODING_TYPE'] = np.where((df_adapted['TABLE_NAME'] == table_name) & (df_adapted['COLUMN_NAME'] == column_name),
                                       encoding, df_adapted['ENCODING_TYPE'])
                df_adapted['VECTOR_COMPRESSION_TYPE'] = np.where((df_adapted['TABLE_NAME'] == table_name) & (df_adapted['COLUMN_NAME'] == column_name),
                                                                 vector_compression, df_adapted['VECTOR_COMPRESSION_TYPE'])

            if len(df_adapted) == 0:
                continue

            if operator_name == 'table_scan':
                new_predictable = feature_preparation_helpers.featurize_table_scans(df_adapted)
            elif operator_name == 'aggregate':
                new_predictable = feature_preparation_helpers.featurize_aggregates(df_adapted)
            elif operator_name == 'projection':
                new_predictable = feature_preparation_helpers.featurize_projections(df_adapted)
            else:
                exit(f"Unknown operator: {operator_name}")


            # For later processing, we set the table and column name to the currently evaluated setting. Since the
            # initial input data frame might include temporary columns, these will occur here as well and they will
            # actually be not relevant at all.
            # It's just hard to filter them out afterwards (the change is 0.0 anyways, so that's fine).
            new_predictable['TABLE_NAME'] = table_name
            new_predictable['COLUMN_NAME'] = column_name
            new_predictable['ENCODING_TYPE'] = encoding
            new_predictable['VECTOR_COMPRESSION_TYPE'] = vector_compression

            df_projected = new_predictable[encoding_selection_constants.PREDICTION_COLUMNS[operator_name]]
            df_projected = df_projected.drop(['execution_time_ms'], axis=1)
            predictions = loaded_model.predict(df_projected)
            predictions = encoding_selection_helpers.adapt_predictions(model, predictions)

            new_predictable['adapted_prediction'] = predictions
            encoding_change_predictions = pd.concat([encoding_change_predictions, new_predictable[['QUERY_HASH',
                                                                                                   'OPERATOR_HASH',
                                                                                                   'TABLE_NAME',
                                                                                                   'COLUMN_NAME',
                                                                                                   'ENCODING_TYPE',
                                                                                                   'VECTOR_COMPRESSION_TYPE',
                                                                                                   'adapted_prediction']]])

    print('')

    if len(encoding_change_predictions) > 0:
        encoding_change_predictions.VECTOR_COMPRESSION_TYPE = encoding_change_predictions.VECTOR_COMPRESSION_TYPE.replace("", np.nan)
        encoding_selection_helpers.adapt_negative_predictions(encoding_change_predictions, "adapted_prediction")
        agg = encoding_change_predictions.reset_index(drop=True).groupby(['QUERY_HASH',
                                                                          'OPERATOR_HASH',
                                                                          'TABLE_NAME',
                                                                          'COLUMN_NAME',
                                                                          'ENCODING_TYPE',
                                                                          'VECTOR_COMPRESSION_TYPE'], dropna=False).agg({'adapted_prediction': ['sum']}).droplevel(1, axis=1)

        ret = baseline_execution_times.reset_index().merge(agg.reset_index(), how='left', on=['QUERY_HASH', 'OPERATOR_HASH']).reset_index(drop=True)
        ret['change'] = ret.adapted_prediction - ret.prediction

        if len(ret.query('ENCODING_TYPE == "Dictionary" and VECTOR_COMPRESSION_TYPE == "FixedSize2ByteAligned" and abs(change) > 1e-6')) > 0:
            print(ret.query('ENCODING_TYPE == "Dictionary" and VECTOR_COMPRESSION_TYPE == "FixedSize2ByteAligned" and abs(change) > 1e-6'))
        assert len(ret.query('ENCODING_TYPE == "Dictionary" and VECTOR_COMPRESSION_TYPE == "FixedSize2ByteAligned" and abs(change) > 1e-6')) == 0
        return ret

    else:
        sys.exit("Unexpected case.")

    return baseline_execution_times


#####
#####
#####       JOINS
#####
#####

# Determines for every query (query hash) the runtime for the baseline encoding (i.e., DictionaryFWI).
# Returns a dataframe that stores the estimated query runtime for the baseline encoding
def predict_join_baselines(df, runtime_models_folder, model = 'heteroscedastic', verbose = False):
    initial_join_data_sets = feature_preparation_helpers.featurize_joins(df)

    initial_data = {'hash_join_materialize': initial_join_data_sets[0],
                    'hash_join_remainder': initial_join_data_sets[1],
                    'sort_merge_join': initial_join_data_sets[2]}

    results = pd.DataFrame()

    for join_model in ['hash_join_materialize', 'hash_join_remainder', 'sort_merge_join']:
        predictable = initial_data[join_model]

        if len(predictable) == 0:
            continue

        loaded_model = joblib.load(os.path.join(runtime_models_folder, join_model, f'{model}.joblib'))
        df_projected = predictable[encoding_selection_constants.PREDICTION_COLUMNS[join_model]]
        df_projected = df_projected.drop(['execution_time_ms'], axis=1)

        predictions = loaded_model.predict(df_projected)
        predictions = encoding_selection_helpers.adapt_predictions(model, predictions)

        baseline_execution_time = sum(predictions)
        if join_model == "hash_join_materialize":
            grouped_predictable = predictable.groupby(["QUERY_HASH", "OPERATOR_HASH"]).agg({"BUILD_SIDE_MATERIALIZING_NS": "min",
                                                                                            "PROBE_SIDE_MATERIALIZING_NS": "min"}).reset_index()
            actual_execution_time = sum(grouped_predictable.BUILD_SIDE_MATERIALIZING_NS + grouped_predictable.PROBE_SIDE_MATERIALIZING_NS)
            actual_execution_time = actual_execution_time / 1000 / 1000
        elif join_model == "hash_join_remainder":
            actual_execution_time = 0.0
            for column in [column for column in predictable.columns if column.endswith("_NS")]:
                if "_MATERIALIZING_" in column:
                    continue
                actual_execution_time += sum(predictable[column])
            actual_execution_time = actual_execution_time / 1000 / 1000
        elif join_model == "sort_merge_join":
            actual_execution_time = sum(predictable['execution_time_ms'])

        # Map all the runtimes back to the columns.
        predictable.loc[:, 'prediction'] = predictions
        predictable['JOIN_MODEL_TYPE'] = join_model

        if verbose: print(f'Join::Error ({join_model}): {(actual_execution_time - baseline_execution_time) / actual_execution_time:.2%} '
                          f'(predicted for baseline: {baseline_execution_time:,.2f}, actual: {actual_execution_time:,.2f})')

        if (actual_execution_time * 2.0 < baseline_execution_time) or (actual_execution_time > 2.0 * baseline_execution_time):
            error = (actual_execution_time - baseline_execution_time) / actual_execution_time
            print(f"WARNING: prediction model '{model}' ({join_model} yielded a large error: {error:.2%} " \
                  f"(predicted for baseline: {baseline_execution_time:,.2f} ms, actual: {actual_execution_time:,.2f} ms).")

        encoding_selection_helpers.adapt_negative_predictions(predictable, "prediction")

        predictable["prediction_error"] = predictable.prediction - predictable.execution_time_ms

        results = results.append(predictable[['QUERY_HASH', 'OPERATOR_HASH', 'JOIN_MODEL_TYPE', 'materialize_side', 'prediction']])

    return results


# Join processing is split in three parts. Two parts of the hash join and one part covers the sort-merge join.
# The hash join is split in two phases:
#   - materialization: we extract two rows from an input join. These rows reflect both materialized columns. This is the only
#                      part where encoding matter as the input is materialized afterwards.
#   - remainder: for the remaining steps of the hash join, we take the input row of the join and simply add a few features to it.
#                That's very similar to what we do for the sort-merge join.
def collect_join_runtimes_per_column(operator_data_folder, runtime_models_folder, model = 'heteroscedastic', verbose = False):
    df = get_initial_operator_data('join', operator_data_folder)
    baseline_execution_times = predict_join_baselines(df, runtime_models_folder, model, verbose)

    loaded_models = {'hash_join_materialize': joblib.load(os.path.join(runtime_models_folder, 'hash_join_materialize', f'{model}.joblib')),
                     'hash_join_remainder': joblib.load(os.path.join(runtime_models_folder, 'hash_join_remainder', f'{model}.joblib'))}

    if Path(os.path.join(runtime_models_folder, 'sort_merge_join', f'{model}.joblib')).exists():
        loaded_models['sort_merge_join'] = joblib.load(os.path.join(runtime_models_folder, 'sort_merge_join', f'{model}.joblib'))

    encoding_change_predictions = pd.DataFrame()

    candidates_left = df[["LEFT_TABLE_NAME", "LEFT_COLUMN_NAME", "DATA_TYPE"]]
    candidates_left.columns = ["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"]
    candidates_right = df[["RIGHT_TABLE_NAME", "RIGHT_COLUMN_NAME", "DATA_TYPE_RIGHT"]]
    candidates_right.columns = ["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"]
    candidates = candidates_left.append(candidates_right)

    # Drop publicates, but keep na() as they mark materialized columns
    candidates = candidates.drop_duplicates()

    print(f"Processing {len(candidates)} change candidates ", end='')
    for row in candidates.itertuples(index=False):
        print('.', end='',flush=True)
        table_name = row.TABLE_NAME
        column_name = row.COLUMN_NAME
        data_type = row.DATA_TYPE
        for encoding, vector_compression in ENCODING_TUPLES:
            segment_encoding_spec_df = pd.DataFrame({'ENCODING_TYPE': [encoding], 'VECTOR_COMPRESSION_TYPE': [vector_compression]})
            segment_encoding_spec_str = encoding_selection_helpers.add_segment_encoding_spec_column(segment_encoding_spec_df).loc[0]['segment_encoding_spec']
            if data_type not in encoding_selection_constants.SUPPORTED_DATA_TYPES[int(encoding_selection_constants.EncodingType[segment_encoding_spec_str].value)]:
                continue

            df_adapted = df.query('(LEFT_TABLE_NAME == @table_name and LEFT_COLUMN_NAME == @column_name) or \
                                   (RIGHT_TABLE_NAME == @table_name and RIGHT_COLUMN_NAME == @column_name)').copy()
            for side, suffix in [("LEFT", ""), ("RIGHT", "_RIGHT")]:
                df_adapted[f'ENCODING_TYPE{suffix}'] = np.where((df_adapted[f'{side}_TABLE_NAME'] == table_name) & (df_adapted[f'{side}_COLUMN_NAME'] == column_name),
                                                       encoding, df_adapted[f'ENCODING_TYPE{suffix}'])
                df_adapted[f'VECTOR_COMPRESSION_TYPE{suffix}'] = np.where((df_adapted[f'{side}_TABLE_NAME'] == table_name) & (df_adapted[f'{side}_COLUMN_NAME'] == column_name),
                                                                 vector_compression, df_adapted[f'VECTOR_COMPRESSION_TYPE{suffix}'])

            for key, featurized in zip(['hash_join_materialize', 'hash_join_remainder', 'sort_merge_join'],
                                       feature_preparation_helpers.featurize_joins(df_adapted)):
                if len(featurized) == 0:
                    continue

                featurized['ENCODING_TYPE'] = encoding
                featurized['VECTOR_COMPRESSION_TYPE'] = vector_compression
                featurized['JOIN_MODEL_TYPE'] = key

                if key == "hash_join_materialize":
                    # in case of materialization, we have two rows each having a table name and a column name. Since
                    # one of the rows might reference a column we are currently not testing, we filter.
                    featurized = featurized.query("TABLE_NAME == @table_name and COLUMN_NAME == @column_name").copy()
                else:
                    # for the non-materialize steps, we have a single row with added features. To assign the runtime
                    # changes to the currently changed encoded column, we add both for the later groupby-aggregation.
                    featurized["TABLE_NAME"] = table_name
                    featurized["COLUMN_NAME"] = column_name


                df_projected = featurized[encoding_selection_constants.PREDICTION_COLUMNS[key]]
                df_projected = df_projected.drop(['execution_time_ms'], axis=1)
                predictions = loaded_models[key].predict(df_projected)
                predictions = encoding_selection_helpers.adapt_predictions(model, predictions)

                featurized.loc[:, 'adapted_prediction'] = predictions

                encoding_change_predictions = pd.concat([encoding_change_predictions, featurized[['QUERY_HASH',
                                                                                                  'OPERATOR_HASH',
                                                                                                  'TABLE_NAME',
                                                                                                  'COLUMN_NAME',
                                                                                                  'ENCODING_TYPE',
                                                                                                  'VECTOR_COMPRESSION_TYPE',
                                                                                                  'JOIN_MODEL_TYPE',
                                                                                                  'materialize_side',
                                                                                                  'adapted_prediction']]])

    print('')

    # The hash join remainder is the estimation for the actual join (after everything is materialized). At this
    # point, there are not encodings anymore.
    # To store the predicted runtime for the later sum up, we only keep one row for DictFWI and remove the table and
    # column information (otherwise, we have the prediction twice, once for left, once for right).
    encoding_change_predictions_extract = encoding_change_predictions.query("JOIN_MODEL_TYPE == 'hash_join_remainder' and " \
                                                                            "ENCODING_TYPE == 'Dictionary' and " \
                                                                            "VECTOR_COMPRESSION_TYPE == 'FixedSize2ByteAligned'").copy()
    encoding_change_predictions = encoding_change_predictions.query("JOIN_MODEL_TYPE != 'hash_join_remainder'")
    encoding_change_predictions_extract_grouped = encoding_change_predictions_extract.groupby(['QUERY_HASH',
                                            'OPERATOR_HASH',
                                            'ENCODING_TYPE',
                                            'VECTOR_COMPRESSION_TYPE',
                                            'JOIN_MODEL_TYPE',
                                            'materialize_side'], dropna=False).agg({'adapted_prediction': 'min'}).reset_index()
    encoding_change_predictions = encoding_change_predictions.append(encoding_change_predictions_extract_grouped)

    if len(encoding_change_predictions) > 0:
        encoding_change_predictions.VECTOR_COMPRESSION_TYPE = encoding_change_predictions.VECTOR_COMPRESSION_TYPE.replace("", np.nan)
        encoding_selection_helpers.adapt_negative_predictions(encoding_change_predictions, "adapted_prediction")

        # left join here, because the initial data might have NULL table/columns for materialized segments. These are
        # not considered as "encoding change candidates" and are thus not part of the "right side" of the join.
        joined = baseline_execution_times.merge(encoding_change_predictions.reset_index(drop=True),
                                                              how='left', on=['QUERY_HASH', 'OPERATOR_HASH', 'JOIN_MODEL_TYPE', 'materialize_side']).reset_index()
        joined['change'] = joined.adapted_prediction - joined.prediction

        ret = joined.reset_index().groupby(['QUERY_HASH',
                                            'OPERATOR_HASH',
                                            'TABLE_NAME',
                                            'COLUMN_NAME',
                                            'ENCODING_TYPE',
                                            'VECTOR_COMPRESSION_TYPE',
                                            'JOIN_MODEL_TYPE',
                                            'materialize_side'], dropna=False).agg({'prediction': 'min',
                                                                                   # 'adapted_prediction': 'sum',   ... # counldn't get it to work :(
                                                                                   'change': 'sum'}).reset_index()

        assert len(ret.query('TABLE_NAME.notna() and ENCODING_TYPE == "Dictionary" and '\
                             'VECTOR_COMPRESSION_TYPE == "FixedSize2ByteAligned" and abs(change) > 1e-6', engine="python")) == 0
        return ret

    return baseline_execution_times.reset_index()

