#!/usr/bin/python3

# Pandas 1.4 causes future warning spamming in XGBoost: disable for now
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Patch scikit learn with Intel Ex (does not work on our evaluation machine; disabled for now)
# from sklearnex import patch_sklearn
# patch_sklearn()

import datetime
import json
import lzma
import math
import matplotlib.pyplot as plt
import multiprocess
import numpy as np
import os
import pandas as pd
import random
import sys
import xgboost as xgb

from joblib import dump
from pathlib import Path
from scipy import optimize
from sklearn import metrics
from sklearn import neural_network
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from time import time
from timeit import default_timer as timer

from helpers import encoding_selection_constants
from helpers import encoding_selection_helpers
from helpers import dag_traversal_helpers
from helpers import feature_preparation_helpers
from helpers import size_estimation


PERFORM_ALLR = True


LINEAR_TREE_JOBS = 8
LINEAR_TREE_MIN_SAMPLES_SPLIT = 0.02
LINEAR_TREE_MIN_SAMPLES_LEAF = 0.01

PROCESS_COUNT = multiprocess.cpu_count()
CALIBRATION_DIR = "calibration"
OPERATOR_FOLDER = "processed_for_calibration_operators"


def load_csv_and_process_dags(folder):
    workload_operators = dag_traversal_helpers.load_and_process_calibration_files(folder, True)
    dag_traversal_helpers.write_results_to_disk(workload_operators, os.path.join(folder, OPERATOR_FOLDER), True)


def prepare_learning_data(folder):
    folder = os.path.join(folder, "processed_for_calibration_operators")
    print(f"{datetime.datetime.now()}: Starting featurization for {folder}.")
    feature_preparation_helpers.featurize_all(folder, folder, PROCESS_COUNT - 2, True)
    print(f"{datetime.datetime.now()}: Featurization done.")


# TODO: settle on how to handle k_fold_evaluation.
def learn_runtime_models(calibration_path, split_config=None, k_fold_evaluation=False, single_model=None):
    if split_config is not None:
        assert len(split_config) == 2, "Unexpected split config passed"
        split_value = split_config[0]
        split_run_id = split_config[1]
        assert split_value <= 1.0 and split_value >= 0.0

    calibration_folder = Path(calibration_path).name
    calibration_run = Path(calibration_path).parent.name
    operator_folder = os.path.join(calibration_path, OPERATOR_FOLDER)


    def prepare_and_split_learning_data(df_orig):
        X = df_orig.drop('execution_time_ms', axis = 1)
        y = df_orig['execution_time_ms']

        if split_config is None:
            yield (X, X, y, y)
            return

        if split_config is not None and not k_fold_evaluation:
            yield train_test_split(X, y, test_size=split_value, random_state=4217*split_run_id)
            return

        k = int(round(1 / split_value))
        kfold = KFold(k, shuffle=True, random_state=17)
        print(f"Running k-fold analysis with k={k}.")
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = y[train_index], y[test_index]

            yield (X_train, X_test, y_train, y_test)


    def linear_regression_ols(X_train, y_train):
        reg = LinearRegression().fit(X_train, y_train)
        return reg


    def xgb_regressor(X_train, y_train):
        reg = xgb.XGBRegressor().fit(X_train, y_train)
        return reg


    class LinearRegressionWrapper:
        def __init__(self, log_transform_dependent=False):
            self.regression = LinearRegression()

        def fit(self, X, y):
            return self.regression.fit(X, y)

        def predict(self, X):
            return self.regression.predict(X)


    ## Least squares percentage regression (without percentages)
    class AdaptedRegression():
        def __init__(self):
            self.regression = LinearRegression(fit_intercept = False)

        def fit(self, X, y):
            X_adapted = X.div(y, axis=0)
            y_adapted = y.div(y, axis=0)

            return self.regression.fit(X_adapted, y_adapted)

        def predict(self, X):
            return self.regression.predict(X)


    class XGBRegressorWrapper:
        def __init__(self, log_transform_dependent=False):
            # pseudohuber is quite nice, but it's much worse at MSE than simple linear regression and still worse
            # overall for all other benchmarks. Listing both in the paper is too much. Unsure.
            # (objective="reg:pseudohubererror", 
            self.regression = xgb.XGBRegressor(n_estimators=200, max_depth=7, eta=0.2)
            self.log_transform_dependent = log_transform_dependent

        def fit(self, X, y):
            if self.log_transform_dependent:
                y_logged = np.log(y)
                return self.regression.fit(X, y_logged)
            return self.regression.fit(X, y)

        def predict(self, X):
            if self.log_transform_dependent:
                predictions = self.regression.predict(X)
                return np.exp(predictions)
            return self.regression.predict(X)


    def adapt_lz4_and_fsst(name):
        return name.replace('LZ4BP', 'LZ4').replace('FSSTBP', 'FSST')


    def adapt_encoding_names(names):
        names = [adapt_lz4_and_fsst(name) for name in names]
        return pd.unique(names)


    MODELS = {
          "LinReg": {"model": LinearRegressionWrapper(), "name": "Linear Regression (OLS)", "active": True, "plot": True, "dump": True},
          "AdaptedLinReg": {"model": AdaptedRegression(), "name": "Linear Regression (adapted)", "active": True, "plot": True, "dump": True},
          "XGBoostReg": {"model": XGBRegressorWrapper(), "name": "XGBoost Regressor", "active": True, "plot": True, "dump": True}}

    if single_model is not None:
        print(f"WARNING: using only a single model: {single_model}")
        for name in MODELS.keys():
            MODELS[name]["active"] = False
        MODELS[single_model]["active"] = True


    # The following code can be shared for all remaining operators as we don't split by encoding and data type
    def non_splitting_models(operator_name, learning_data):
        output_model_dir = os.path.join(models_runtime_folder, operator_name)
        os.makedirs(output_model_dir, exist_ok = True)

        output_error_metrics_dir = os.path.join(error_metrics_results_folder, operator_name)
        os.makedirs(output_error_metrics_dir, exist_ok = True)

        training_data_dir = os.path.join(output_error_metrics_dir, "training_data")
        os.makedirs(training_data_dir, exist_ok = True)

        model_runtimes = pd.DataFrame()

        learning_data = learning_data[encoding_selection_constants.PREDICTION_COLUMNS[operator_name]]
        for X_train, X_test, y_train, y_test in prepare_and_split_learning_data(learning_data):
            test_results = X_test.copy()
            test_results.loc[:,'actual'] = y_test

            assert len(test_results.query("actual < 0")) == 0

            if split_config is None:
                print(f"{datetime.datetime.now()}: Measuring training and prediction runtimes on full data set (no splitting: ", end="")
            else:
                print(f"{datetime.datetime.now()}: Measuring training and prediction runtimes on split data set (", end="")
            print(f"{len(X_train):,} obs. in train set, {len(X_test):,} obs. in test set):")
            for model_shortname, model_dict in MODELS.items():
                if not model_dict["active"]:
                    continue

                print(model_shortname, end="")
                fit_start = timer()
                fitted_model = model_dict["model"].fit(X_train, y_train)
                fit_end = timer()

                predict_start = timer()
                predictions = model_dict["model"].predict(X_test)
                predict_end = timer()

                predictions = encoding_selection_helpers.adapt_predictions(model_shortname, predictions)
                test_results.loc[:, f"prediction_{model_shortname}"] = predictions
                encoding_selection_helpers.adapt_negative_predictions(test_results, f"prediction_{model_shortname}")

                dump(fitted_model, os.path.join(output_model_dir, f"{model_shortname}.joblib"))

                print(f" (train: {fit_end-fit_start:.4f} s, test: {predict_end-predict_start:.4f} s), ", end="", flush=True)
                test_train_ratio = len(X_test) / (len(X_test) + len(X_train))
                model_runtimes = pd.concat([model_runtimes, pd.DataFrame([{"MODEL": model_shortname, "OPERATOR": operator_name,
                                                         "MEASUREMENT": "TRAIN_RUNTIME_MS",
                                                         "VALUE": (fit_end - fit_start) * 1000,
                                                         "TEST_TRAIN_RATIO": test_train_ratio},
                                                        {"MODEL": model_shortname, "OPERATOR": operator_name,
                                                         "MEASUREMENT": "TEST_RUNTIME_MS",
                                                         "VALUE": (predict_end - predict_start) * 1000,
                                                         "TEST_TRAIN_RATIO": test_train_ratio},
                                                        {"MODEL": model_shortname, "OPERATOR": operator_name,
                                                         "MEASUREMENT": "TRAIN_OBSERVATIONS",
                                                         "VALUE": len(X_train), "TEST_TRAIN_RATIO": test_train_ratio},
                                                        {"MODEL": model_shortname, "OPERATOR": operator_name,
                                                         "MEASUREMENT": "TEST_OBSERVATIONS",
                                                         "VALUE": len(X_test), "TEST_TRAIN_RATIO": test_train_ratio}])])

            print("")

            encoding_selection_helpers.evaluate_and_plot(test_results, X_train, y_train, MODELS, True,
                                                            os.path.join(output_error_metrics_dir,
                                                                             f"{operator_name}.pdf"), False)

        model_runtimes.to_csv(os.path.join(output_model_dir, f"model_measurements_{int(time())}.csv"))

    workload_models_directory = f"generated_models/{calibration_run}"
    if split_config is not None:
        workload_models_directory += f"__split_{str(split_value).replace('.', '_')}_r{split_run_id}"
    runtime_folder = os.path.join(workload_models_directory, "runtime")
    models_runtime_folder = os.path.join(runtime_folder, "models")
    error_metrics_results_folder = os.path.join(runtime_folder, "error_metrics")
    prediction_results_folder = os.path.join(runtime_folder, "prediction_results")

    os.makedirs(models_runtime_folder, exist_ok = True)
    os.makedirs(error_metrics_results_folder, exist_ok = True)
    os.makedirs(prediction_results_folder, exist_ok = True)

    #### table scans
    table_scans = pd.read_csv(os.path.join(operator_folder, "table_scan_prepared.csv.bz2"), low_memory=False)
    print(f'{datetime.datetime.now()}: Table scan data contains {len(table_scans):,} rows.')
    non_splitting_models("table_scan", table_scans)

    #### aggregates
    aggregates = pd.read_csv(os.path.join(operator_folder, "aggregate_prepared.csv.bz2"), low_memory=False)
    print(f'{datetime.datetime.now()}: Aggregate data contains {len(aggregates):,} rows.')
    non_splitting_models("aggregate", aggregates)

    ### joins
    hash_join_materialize = pd.read_csv(os.path.join(operator_folder, "hash_join_materialize_prepared.csv.bz2"), low_memory=False)
    print(f'{datetime.datetime.now()}: Hash join (materialize) data contains {len(hash_join_materialize):,} rows.')
    non_splitting_models("hash_join_materialize", hash_join_materialize)

    hash_join_remainder = pd.read_csv(os.path.join(operator_folder, "hash_join_remainder_prepared.csv.bz2"), low_memory=False)
    print(f'{datetime.datetime.now()}: Hash join (remainder) data contains {len(hash_join_remainder):,} rows.')
    non_splitting_models("hash_join_remainder", hash_join_remainder)

    sort_merge_joins = pd.read_csv(os.path.join(operator_folder, "sort_merge_join_prepared.csv.bz2"), low_memory=False)
    if len(sort_merge_joins) > 0:
        print(f'{datetime.datetime.now()}: Sort-merge join data contains {len(sort_merge_joins):,} rows.')
        non_splitting_models("sort_merge_join", sort_merge_joins)

    #### projections
    projections_file = os.path.join(operator_folder, "projection_prepared.csv.bz2")
    if Path(projections_file).exists():
        projections = pd.read_csv(projections_file, low_memory=False)
        print(f'{datetime.datetime.now()}: Projection data contains {len(projections):,} rows.')
        non_splitting_models("projection", projections)

    if split_config is not None:
        with open(os.path.join(workload_models_directory, "split_config.json"), "w") as split_config_file:
            json.dump({"split_value": split_value, "split_run_id": split_run_id}, split_config_file)


def learn_size_models(workloads_folder, split_config=None):
    if split_config is not None:
        assert len(split_config) == 2, "Unexpected split config passed"
        split_value = split_config[0]
        split_run_id = split_config[1]
        assert split_value <= 1.0 and split_value >= 0.0

    calibration_folder = Path(workloads_folder).name
    calibration_run = Path(workloads_folder).parent.name
    if split_config is not None:
        calibration_run += f"__split_{str(split_value).replace('.', '_')}_r{split_run_id}"

    print("WARNING: size models are currently not split.")
    results_folder = os.path.join("generated_models", calibration_run, "size")
    size_estimation.create_size_model_learning_data(workloads_folder, results_folder)
    size_estimation.learn_and_store_size_models(results_folder, results_folder, False)

