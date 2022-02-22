#!/usr/bin/env python3

# Pandas 1.4 causes future warning spamming in XGBoost: disable for now
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import os

import numpy as np
import pandas as pd
import sklearn.metrics
import sys
import xgboost as xgb

from joblib import dump
from sklearn.linear_model import LinearRegression

sys.path.append("..")
from helpers import encoding_selection_helpers


def create_size_model_learning_data(read_folder, write_folder):
    os.makedirs(write_folder, exist_ok = True)
    size_data_list = []

    for path, d, f in os.walk(read_folder):
        for file in f:
            if file == 'meta_data_segment.csv':
                segment_meta_data = pd.read_csv(os.path.join(path, file), low_memory=False)
                data_characteristics = pd.read_csv(os.path.join(path, "..", "data_characteristics.csv"), keep_default_na=False, na_values=['NULL'], low_memory=False)
                chunk_meta_data = pd.read_csv(os.path.join(path, "meta_data_chunk.csv"), low_memory=False) # segment sizes
                column_meta_data = pd.read_csv(os.path.join(path, "meta_data_column.csv"), low_memory=False) # is nullable
                segment_chunk_meta_data = pd.merge(segment_meta_data, chunk_meta_data, how="inner",
                                                   on=['TABLE_NAME', 'CHUNK_ID'], suffixes=('', '_CHUNK'))
                segment_chunk_column_meta_data = pd.merge(segment_chunk_meta_data, column_meta_data,
                                                                           how="inner", on=['TABLE_NAME', 'COLUMN_NAME'],
                                                                           suffixes=('', '_COL'))
                size_data_list.append(pd.merge(segment_chunk_column_meta_data, data_characteristics,
                                                      how="inner", on=['TABLE_NAME', 'COLUMN_ID', 'CHUNK_ID'],
                                                      suffixes=('', '_DATACHAR')))

    size_data = pd.concat(size_data_list)

    encoding_selection_helpers.add_segment_encoding_spec_column(size_data)

    # Are NULL for non-string columns
    size_data['VALUE_SWITCHES'] = size_data['VALUE_SWITCHES'].replace(np.nan, 0.0)
    size_data['AVG_STRING_LENGTH'] = size_data['AVG_STRING_LENGTH'].replace(np.nan, 0.0)
    size_data['MAX_STRING_LENGTH'] = size_data['MAX_STRING_LENGTH'].replace(np.nan, 0.0)
    size_data['max_chars'] = size_data['MAX_STRING_LENGTH'] * size_data['ROW_COUNT']

    size_data['row_count_log'] = np.log(size_data['ROW_COUNT'] + 1e-9)
    size_data['row_count_log'] = size_data['row_count_log'].replace(np.nan, 0.0)

    size_data['nullable_rows'] = size_data['ROW_COUNT'] * size_data['NULLABLE']

    size_data['segment_av_width'] = np.minimum(32, np.maximum(8.0, np.power(2, np.ceil(np.log2(size_data['DISTINCT_VALUE_COUNT'] + 1e-9)))))
    size_data['segment_av_width'] = size_data['segment_av_width'].replace(np.nan, 0.0)
    size_data['segment_av_width'] = np.where(size_data['ENCODING_TYPE'] == 'Dictionary', size_data['segment_av_width'], 0.0)

    # only neccesary until we fixed the size info creation
    size_data['segment_encoding_spec'] = np.where(size_data['segment_encoding_spec'] == 'lz4', 'lz4_bp',
                                                  size_data['segment_encoding_spec'])
    
    size_data_file = os.path.join(write_folder, 'size_model_prepared.csv.bz2')
    size_data.to_csv(size_data_file, sep = ',', index = False)


def learn_and_store_size_models(read_folder, write_folder, verbose = False):
    size_model_data = pd.read_csv(os.path.join(read_folder, 'size_model_prepared.csv.bz2'))

    with open(os.path.join(write_folder, 'size__models.csv'), 'w') as file:
        file.write(f'segment_encoding_spec,data_type,intercept,row_count,estimated_distinct_values,nullable_rows,VALUE_SWITCHES,AVG_STRING_LENGTH,MAX_STRING_LENGTH\n')
        for encoding in pd.unique(size_model_data['segment_encoding_spec']):
            for data_type in pd.unique(size_model_data['COLUMN_DATA_TYPE']):
                model_data = size_model_data.query('segment_encoding_spec == @encoding and COLUMN_DATA_TYPE == @data_type')

                if len(model_data) == 0:
                    continue

                output_folder = f'{write_folder}/{encoding}_{data_type}'
                os.makedirs(output_folder, exist_ok = True)

                X_train = model_data[['DISTINCT_VALUE_COUNT', 'ROW_COUNT', 'row_count_log', 'segment_av_width',
                                      'nullable_rows', 'VALUE_SWITCHES', 'AVG_STRING_LENGTH', 'MAX_STRING_LENGTH']]
                y_train = model_data['SIZE_IN_BYTES']

                reg = LinearRegression().fit(X_train, y_train)

                coefficients = ','.join(map(str, reg.coef_))
                if verbose: file.write(f'{encoding},{data_type},{reg.intercept_},{coefficients}\n')
                dump(reg, f'{output_folder}/ols_reg.joblib')

                predicted = reg.predict(X_train)
                errors = abs((predicted - y_train) / y_train)
                if verbose: print(f'LR:  {encoding} & {data_type}: {np.mean(errors)}')

                reg = xgb.XGBRegressor().fit(X_train, y_train, eval_metric='rmse')
                dump(reg, f'{output_folder}/xgb_reg.joblib')

                predicted = reg.predict(X_train)
                errors = abs((predicted - y_train) / y_train)
                if verbose: print(f'XGB: {encoding} & {data_type}: {np.mean(errors)}')

