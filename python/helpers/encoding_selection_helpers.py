#!/usr/bin/env python3

import datetime
import json
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from numpy.testing import assert_almost_equal
from sklearn.model_selection import KFold

sys.path.append("..")
from helpers import encoding_selection_constants

replacements = {"FixedW": "fwi", "BitPac": "bp"}


# For Q-Error and other metrics, remove zeros
def remove_zeros(v1, v2):
    assert len(v1) == len(v2)
    mask = (v1 == 0.0) | (v2 == 0.0)
    v1_adapted = v1[~mask]
    v2_adapted = v2[~mask]

    return (v1_adapted, v2_adapted)


def add_segment_encoding_spec_column(df):
    df['ENCODING_TYPE'].fillna('unencoded', inplace=True)
    df['ENCODING_TYPE_lower'] = df['ENCODING_TYPE'].str.lower()
    df['VECTOR_COMPRESSION_TYPE'].fillna('', inplace=True)
    acronyms = ["_" + replacements[x[:6]] if x[:6] in replacements else '' for x in df[f'VECTOR_COMPRESSION_TYPE']]
    df['segment_encoding_spec'] = df['ENCODING_TYPE_lower'] + acronyms

    return df


def get_segment_encoding_spec(encoding, vector_compression):
    if isinstance(vector_compression, str):
        return f"{encoding.lower()}_{replacements[vector_compression[:6]]}"

    assert np.isnan(vector_compression)
    return encoding.lower()


def mse(y_test, y_pred):
    assert len(y_test) == len(y_pred), "Actuals and predictions differ in length."
    return np.mean(np.power(y_test - y_pred, 2))


def rmse(y_test, y_pred):
    return np.sqrt(mse(y_test, y_pred))


def mare(y_test, y_pred):
    assert len(y_test) == len(y_pred), "Actuals and predictions differ in length."
    return np.mean(np.absolute((y_pred - y_test) / y_test))


def mape(y_test, y_pred):
    assert len(y_test) == len(y_pred), "Actuals and predictions differ in length."
    return 100 * mare(y_test, y_pred)


def mspe(y_test, y_pred):
    assert len(y_test) == len(y_pred), "Actuals and predictions differ in length."
    return np.mean(np.power(((y_pred - y_test) / y_test), 2))

    
def smape_the_third(y_test, y_pred): #  https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    assert len(y_test) == len(y_pred), "Actuals and predictions differ in length."

    difference = y_pred - y_test
    abs_difference = np.abs(difference)
    denominator = np.sum(y_pred + y_test)
    assert denominator != 0

    return np.sum(abs_difference) / denominator


def average_relative_error(y_test, y_pred):
    assert len(y_test) == len(y_pred), "Actuals and predictions differ in length."
    y_test2, y_pred2 = remove_zeros(y_test, y_pred)
    if len(y_test) != len(y_test2):
        print(f"WARNING: removed {len(y_test) - len(y_test2)} (of {len(y_test)}) zeros for average_relative_error metric.")
    return np.mean(np.absolute(y_test2 - y_pred2) / y_test2)


def average_absolute_error(y_test, y_pred):
    assert len(y_test) == len(y_pred), "Actuals and predictions differ in length."
    return np.mean(np.absolute(y_test - y_pred))


def q_error(y_test, y_pred):
    assert len(y_test) == len(y_pred), "Actuals and predictions differ in length."
    y_test2, y_pred2 = remove_zeros(y_test, y_pred)
    if len(y_test) != len(y_test2):
        print(f"WARNING: removed {len(y_test) - len(y_test2)} (of {len(y_test)}) zeros for q_error metric.")
    return np.mean(np.maximum(np.divide(y_test2, y_pred2), np.divide(y_pred2, y_test2)))


# A Better Measure of Relative Prediction Accuracy for Model Selection and Model Estimation
# Journal of the Operational Research Society (2015) 66, 1352–1362
def log_of_the_accuracy_ratio(y_test, y_pred):
    assert len(y_test) == len(y_pred), "Actuals and predictions differ in length."
    assert not np.any(y_test < 0)

    y_test2, y_pred2 = remove_zeros(y_test, y_pred)
    y_test2, y_pred2 = remove_zeros(y_test, y_pred)
    if len(y_test) != len(y_test2):
        print(f"WARNING: removed {len(y_test) - len(y_test2)} (of {len(y_test)}) zeros for LogQ metric.")

    return np.sum(np.power(np.log(np.absolute(y_pred2 / y_test2)), 2))


def evaluate_and_plot(df, X, y, models, skip_plot = False, plot_file_name = None, print_latex_table = False,
                      test_size = 1.0, k = 1):
    df_summary = pd.DataFrame(columns=['Metric', "Linear Regression (OLS)", "Linear Regression (adapted)"])
    df_plotting = pd.DataFrame(columns=['Error Metric', 'Approach', 'Error']) # columns needed?!?!?
    print_legend = True

    # Printing runtimes of models
    # display(df[[col for col in df.columns if 'fitting_' in col]].min())

    # filtering on actual
    actual_filters = {
        '$Q_1$-$Q_4$': '>= 0',
        '$Q_1$, $Q_2$': f"<= {df['actual'].median():8.4f}",
        '$Q_3$, $Q_4$': f"> {df['actual'].median():8.4f}"
    }
    

    def eval_and_append(_title, _result, _metric, _sum_dict, _df_plotting, plot=True):
        _sum_dict[_title] = [_result]
        if plot:
            _df_plotting = pd.concat([_df_plotting, pd.DataFrame({'Error Metric': [_metric], 'Approach': [_title],
                                                                  'Error': [_result]})])
        return (_sum_dict, _df_plotting)


    metrics = [mse,
               rmse,
               mare,
               mape, mspe, smape_the_third,
               log_of_the_accuracy_ratio,
               average_absolute_error,
               average_relative_error,
               q_error]

    for filter_title, filter_str in actual_filters.items():
        f_df = df.query(f'actual {filter_str}')
        nice_names = {"smape_the_third": "SMAPE", "log_of_the_accuracy_ratio": "LogQ",
                      "average_absolute_error": "Avg. abs. err.", "average_relative_error": "Avg. rel. err.",
                      "q_error": "Q-Error"}
        for fun in metrics:
            function_name = nice_names[fun.__name__] if fun.__name__ in nice_names else fun.__name__.upper()
            metric = f'{function_name} ({filter_title})'
            sum_dict = {'Metric': metric}

            for model_shortname, model_dict in models.items():
                if not model_dict["active"]:
                    continue

                sum_dict, df_plotting = eval_and_append(model_dict["name"], fun(f_df["actual"],
                                                        f_df[f"prediction_{model_shortname}"]),
                                                        metric, sum_dict, df_plotting, model_dict["plot"])

            df_summary = pd.concat([df_summary, pd.DataFrame(sum_dict)])

        if filter_title == 'all data points' and not skip_plot: # simplify for now
            plot(f_df, filter_title, print_legend)
            print_legend = False

    tmp = df_plotting['Error Metric'].apply(lambda x: x.split(' ('))
    df_plotting.loc[:, 'Metric'] = tmp.apply(lambda x: x[0])
    df_plotting.loc[:, 'Quantile'] = tmp.apply(lambda x: x[1])
    df_plotting['Quantile'] = df_plotting['Quantile'].str.replace("Q_", "", regex=False)
    df_plotting['Quantile'] = df_plotting['Quantile'].str.replace("$", "", regex=False)
    df_plotting['Quantile'] = df_plotting['Quantile'].str.replace(" ", "", regex=False)
    df_plotting['Quantile'] = df_plotting['Quantile'].str.replace("(", "", regex=False)
    df_plotting['Quantile'] = df_plotting['Quantile'].str.replace(")", "", regex=False)

    previous_results_path = Path(plot_file_name).parent / "model_evaluation.csv"
    if previous_results_path.exists():
        df_plotting_previous = pd.read_csv(previous_results_path)
        df_plotting = df_plotting_previous.append(df_plotting, ignore_index=True)
    df_plotting.to_csv(previous_results_path, index=False)


def write_configuration_to_csv(folder, name, configuration, workload):
    with open(f'{folder}/conf__{name}.csv', 'w') as file:
        for t_id, table in enumerate(configuration):
            table_name = workload['table_names'][t_id]
            for chunk_id, chunk in enumerate(table[:workload['table_dimensions'][t_id][0]]):
                for c_id, cell in enumerate(chunk[:workload['table_dimensions'][t_id][1]]):
                    enc_str = encoding_selection_constants.EncodingType(int(cell)).name
                    enc_type_str = enc_str.replace('FWI', '', regex=False).replace('BitPacking', '', regex=False)
                    vector_compression_type_str = 'None'
                    vector_compression_type_str = 'Fixed-width integer' if 'FWI' in enc_str else vector_compression_type_str
                    vector_compression_type_str = 'Bit-packing' if 'BP' in enc_str else vector_compression_type_str
                    column_name = workload['column_meta_data'].loc[(t_id,c_id), 'COLUMN_NAME']
                    file.write(f'{table_name},{column_name},{chunk_id},{enc_type_str},{vector_compression_type_str}\n')

def write_configuration_to_json(configurations_folder, model_name, budget, configuration, runtime, workload, prediction_model_name):
    # Takes e.g. "dictionary_bp" and returns ("Dictionary", "BitPacking") or ("LZ4", None) for "LZ4"
    # We use Hyrise names for the configuration to minimize the code in the plugins.
    def map_segment_encoding_string_to_hyrise(encoding_str):
        map_to_hyrise_name = {"dictionary": "Dictionary",
                              "fixedstringdictionary": "FixedStringDictionary",
                              "lz4": "LZ4",
                              "unencoded": "Unencoded",
                              "runlength": "RunLength",
                              "frameofreference": "FrameOfReference",
                              "fsst": "FSST"}
        encoding_type_str = encoding_str.replace("_fwi", "").replace("_bp", "")
        hyrise_encoding_type_str = map_to_hyrise_name[encoding_type_str]

        if "_fwi" in encoding_str or "_bp" in encoding_str:
            # using Hyrise names, minimizing the code in the plugins
            vector_encoding = "Fixed-width integer" if "_fwi" in encoding_str else "Bit-packing"
            return (hyrise_encoding_type_str, vector_encoding)
        return (hyrise_encoding_type_str, None)

    with open(f"{configurations_folder}/{model_name}_{budget}.json", "w") as json_file:
        json_content = {}
        json_content["configuration"] = {}
        json_configuration = json_content["configuration"]
        for t_id, table in enumerate(configuration):
            table_name = workload['table_names'][t_id]
            json_configuration[table_name] = {}
            for chunk_id, chunk in enumerate(table[:workload['table_dimensions'][t_id][0]]):
                json_configuration[table_name][chunk_id] = {}
                for column_id, cell in enumerate(chunk[:workload['table_dimensions'][t_id][1]]):
                    internal_enc_str = encoding_selection_constants.EncodingType(int(cell)).name
                    hyrise_enc_str = map_segment_encoding_string_to_hyrise(internal_enc_str)
                    json_configuration[table_name][chunk_id][column_id] = {"SegmentEncodingType": hyrise_enc_str[0]}
                    if hyrise_enc_str[1] is not None and hyrise_enc_str[0] != "LZ4":
                        json_configuration[table_name][chunk_id][column_id]["VectorEncodingType"] = hyrise_enc_str[1]

        json_content["context"] = {"budget_in_bytes": budget, "selection_model": model_name, "runtime": runtime,
                                   "timestamp": str(datetime.datetime.now()), "prediction_model": prediction_model_name}

        json.dump(json_content, json_file, indent=2)


def adapt_negative_predictions(df, column_name):
    # Please use with caution, meant for debugging.
    return

    # A scan on TPC-H nation with a few dozens of tuples might change from a prediction of 0.03ms prediction to 110 ms.
    min_pred = min(df[column_name])
    if min_pred >= 0:
        # everything is fine, all values > 0
        return

    max_pred = max(df[column_name])
    if max_pred < 0:
        # shift all values to positive range
        df[column_name] = df[column_name] + (-1.000001 * min_pred)
        print("WARNING: Shifted predictions to positive range.")
        return

    # scale [min, max] to [0, max]
    print("Scaled predictions from [min, max] to [0, max]")
    df[column_name] = 0.000001 + (df[column_name] + (-1 * min_pred)) * (max_pred / (max_pred - min_pred))


def adapt_predictions(model_name, predictions):
    # Be careful. Only use this method in case the model is loaded from file. See object LinearRegressionWrapper().
    if "LogY" in model_name:
        predictions = np.exp(predictions - 1)
        if "Poisson" in model_name:
            predictions = predictions / 1_000_000

    return predictions

