#!/usr/bin/env python3

import math
import numpy as np
import os
import pandas as pd
import shutil
import time

from datetime import datetime
from joblib import Parallel, delayed
from multiprocessing import Manager
from pathlib import Path
from timeit import default_timer as timer

from helpers import feature_preparation_helpers
from helpers import encoding_selection_helpers

from selection import column_runtime_change_prediction_helper
from selection import selection_approaches
from selection import workload_loading

WRITE_CSV_CONFIGURATIONS = False
WRITE_JSON_CONFIGURATIONS = True

def get_robustness_constraints_name(robustness_constraints):
    name = "default"
    if robustness_constraints is not None:
        assert robustness_constraints[0] in ["QuerySlowdown", "EqualSlowdown", "EqualRegression"]
        if robustness_constraints[0] == "QuerySlowdown":
            name = "queryslowdown_"
            first = True
            for k, v in robustness_constraints[1].items():
                if not first:
                    name += "___"
                name += k.replace(" ", "_") + "__" + str(v).replace(".", "_")
                first = False
        elif robustness_constraints[0] == "EqualSlowdown":
            equal_slow_down_factor = robustness_constraints[1]
            name = f"equalslowdown{str(equal_slow_down_factor).replace('.', '_')}"
        elif robustness_constraints[0] == "EqualRegression":
            equal_slow_down_factor = robustness_constraints[1]
            name = f"equalregression_{str(equal_slow_down_factor).replace('.', '_')}"

    return name


def run_models_for_budget(pid, budget, budget_start, workload, models_to_evaluate, robustness_constraints,
                          benchmark_configurations_folder, model_to_use, manager):
    results = []  # list of dicts
    query_runtime_results = []
    for model_name, model_probs in models_to_evaluate.items():
        if model_name == "StaticBestRatio" and manager["yielded_static_best_ratio_config"]:
            print("Skipping StaticBestRatio, already yielded a configuration.")
            continue

        start = timer()
        sel = model_probs["model"]

        if robustness_constraints is not None:
            if model_name != "LPCompressionSelection":
                continue
            sel.set_constraint(robustness_constraints)

        if not f"model_max_size__{model_name}" in manager:
            manager[f"model_max_size__{model_name}"] = -17

        # We usually use Gurobi or use the open-source solver HiGHS (unfortunately, not part of the paper
        # evaluation; first release in early 2022). For GH actions, we use Cbc as it comes with PuLP.
        # However, please be aware that Cbc can be slow and has problems with larger problems (e.g.,
        # robust configurations).
        sel.set_solver("Cbc")                                
        sel.set_thread_count(1)

        if "alpha" in model_probs:
            sel.set_alpha(model_probs["alpha"])
        if "weight_function" in model_probs:
            sel.set_weight_function(model_probs["weight_function"])
        fit = timer()
        sel.fit(budget)
        finish = timer()

        result = sel.get_full_result()
        print(f"{datetime.now()}: configuration of {model_name} for budget of {budget:,} ({(budget / budget_start):.0%}): ", end="")


        if result[3] is None:
            print("unsuccessful.")
            continue
        print("successful.")

        if result[2] > budget:
            print(f"WARNING: model {model_name} exceeded the budget of {budget:,} (needed size: {result[2]:,}, overshot by {result[2]-budget:,}).")
            if "Abadi" not in model_name and "Static" not in model_name:
                continue

        robustness_constraints_name = get_robustness_constraints_name(robustness_constraints)

        results.append({"MODEL": model_name, "BUDGET": budget, "SIZE_IN_BYTES": result[2], "CUMULATIVE_RUNTIME_MS": result[1],
                        "APPROACH_LOADING": fit - start, "APPROACH_FITTING": finish - fit, "BUDGET_FEASIBLE": True,
                        "ROBUSTNESS_CONFIGURATION": robustness_constraints_name})

        if model_name == "LPCompressionSelection":
            for query_id, runtime in sel.get_query_runtimes().items():
                query_runtime_results.append({"ROBUSTNESS_CONFIGURATION": robustness_constraints_name,
                                              "BENCHMARK_ITEM_NAME": query_id,
                                              "BENCHMARK_ITEM_RUNTIME": runtime, "BUDGET": budget})

        if result[2] == manager[f"model_max_size__{model_name}"]:
            print(f"Skipping configuration of {model_name} for budget {budget}: next larger budget yielded same size.")
            continue

        manager[f"model_max_size__{model_name}"] = max(result[2], manager[f"model_max_size__{model_name}"])

        configuration_directory = os.path.join(benchmark_configurations_folder, model_name)
        os.makedirs(configuration_directory, exist_ok = True)

        if WRITE_CSV_CONFIGURATIONS:
            encoding_selection_helpers.write_configuration_to_csv(configuration_directory, f'budget__{model_name}', result[3], workload)

        if WRITE_JSON_CONFIGURATIONS:
            encoding_selection_helpers.write_configuration_to_json(configuration_directory, model_name, budget,
                                                                      result[3], result[1], workload, model_to_use)

        if model_name == "StaticBestRatio":
            # if the static model made it up here, it was valid
            manager["yielded_static_best_ratio_config"] = True

    return (results, query_runtime_results)


def run_compression_selection_comparison(short_name, calibration_run, robustness_constraints, workload_folder,
                                         models_folder, model_to_use, budget_steps_stretch_factor):
    processed_for_selection_folder = os.path.join(workload_folder, "..", "processed_for_selection_folder")
    if Path(processed_for_selection_folder).exists():
        print(f"WARNING: {processed_for_selection_folder} already exists.")
        current_timestamp = float(datetime.utcnow().timestamp())
    
        if os.path.getmtime(processed_for_selection_folder) < (current_timestamp - 3600):
            # delete if directory is older than one hour
            try:
                shutil.rmtree(processed_for_selection_folder)
                print("Delete processed_for_selection_folder directory. Creating empty one and continue ...")
                os.makedirs(processed_for_selection_folder)
            except OSError as e:
                print("Could not delete processed_for_selection_folder directory (" + e + "). Continueing ...")

    workload = workload_loading.parse_file_based_workload(workload_folder,
                    os.path.join(models_folder, "runtime", "models"),
                    os.path.join(models_folder, "size"),
                    True, True, True, True, True, model_to_use)
    print(f"Size of minimal possible configuration: {workload['minimal_configuration_size']:,.2f} bytes.")

    robustness_constraints_name = get_robustness_constraints_name(robustness_constraints)

    benchmark_configurations_folder = os.path.join("evaluation", calibration_run, f"configurations__{robustness_constraints_name}", short_name)
    os.makedirs(benchmark_configurations_folder, exist_ok = True)

    budget_start = int(workload['minimal_configuration_size'] * 1.000001)
    budget_max = int(budget_start * 2.5)

    ways_to_obtain_max_buget = ['dictionarysize', 'LPconverging', 'minimal_runtime']
    obtain_max_budget_by = ways_to_obtain_max_buget[2]
    if obtain_max_budget_by == 'dictionarysize':
        budget_max = workload['all_dictionary_size'] * 1.1
        print(f'\nSet max budget to {budget_max/1000/1000:,.4f} MB (min is {budget_start/1000/1000:,.4f} MB, '
              f'Dictionary size is {workload["all_dictionary_size"]/1000/1000:,.4f} MB).')
    elif obtain_max_budget_by == 'minimal_runtime':
        # The backwards heuristics initializes every segment with the fastest encoding.
        # That's our optimum.
        backwards_heuristic = selection_approaches.BackwardsGreedyHeuristic(workload,
                                selection_approaches.weight_by_size_to_runtime_change_ratio,
                                2.0)
        budget_max = backwards_heuristic.initial_configuration_size
        print(f'\nSet max budget to {budget_max/1000/1000:,.4f} MB (minimum is {budget_start/1000/1000:,.4f} MB).')
    else:
        exit("Unexpected.")
    
    initial_budget_diff_percent_widths = [0.1, 0.25, 0.5, 1, 2, 4]
    budget_diff_percent_widths = [budget_steps_stretch_factor * width for width in initial_budget_diff_percent_widths]
    steps_per_budget_diff_percent_widths = [10, 5, 5, 5, 10, int(10e6)]  # we stop as soon as we have passed 100%

    budgets_to_measure = [budget_start]
    last_budget = budget_start
    one_percent_budget_diff = (budget_max - budget_start) * 0.01
    for percent_width, step_count in zip(budget_diff_percent_widths, steps_per_budget_diff_percent_widths):
        for step in range(step_count):
            if last_budget > (budget_max * 1.1):
                break

            budget = last_budget + percent_width * one_percent_budget_diff
            budgets_to_measure.append(int(budget))
            last_budget = budget

    models_to_evaluate = {
                          'GreedyAlpha0.5': {'model': selection_approaches.ForwardsGreedyHeuristic(
                                                          workload,
                                                          selection_approaches.weight_by_runtime_change_to_size_ratio,
                                                          0.5),
                                             'weight_function': selection_approaches.weight_by_runtime_change_to_size_ratio},
                          'GreedyAlpha1.0': {'model': selection_approaches.ForwardsGreedyHeuristic(
                                                          workload,
                                                          selection_approaches.weight_by_runtime_change_to_size_ratio,
                                                          1.0),
                                             'weight_function': selection_approaches.weight_by_runtime_change_to_size_ratio},
                          'GreedyAlpha2.0': {'model': selection_approaches.ForwardsGreedyHeuristic(
                                                          workload,
                                                          selection_approaches.weight_by_runtime_change_to_size_ratio,
                                                          2.0),
                                             'weight_function': selection_approaches.weight_by_runtime_change_to_size_ratio},
                          'GreedyBackwardsAlpha0.5': {'model': selection_approaches.BackwardsGreedyHeuristic(
                                                                   workload,
                                                                   selection_approaches.weight_by_size_to_runtime_change_ratio,
                                                                   0.5),
                                                      'weight_function': selection_approaches.weight_by_size_to_runtime_change_ratio},
                          'GreedyBackwardsAlpha1.0': {'model': selection_approaches.BackwardsGreedyHeuristic(
                                                                   workload,
                                                                   selection_approaches.weight_by_size_to_runtime_change_ratio,
                                                                   1.0),
                                                      'weight_function': selection_approaches.weight_by_size_to_runtime_change_ratio},
                          'GreedyBackwardsAlpha2.0': {'model': selection_approaches.BackwardsGreedyHeuristic(
                                                                   workload,
                                                                   selection_approaches.weight_by_size_to_runtime_change_ratio,
                                                                   2.0),
                                                      'weight_function': selection_approaches.weight_by_size_to_runtime_change_ratio},
                          'LPCompressionSelection': {'model': selection_approaches.LPCompressionSelection(workload, None, False),
                                                     'weight_function': None},
                          'StaticBestRatio': {'model': selection_approaches.StaticBestRatio(workload)}}

    results = pd.DataFrame()
    manager = Manager().dict()
    manager["yielded_static_best_ratio_config"] = False

    #
    # MAIN LOOP
    #
    process_results = Parallel(n_jobs=1)(delayed(run_models_for_budget)(pid, budget, budget_start, workload, models_to_evaluate,
                                                  robustness_constraints, benchmark_configurations_folder, model_to_use, manager)
                                         for pid, budget in enumerate(budgets_to_measure))

    all_results = []
    all_query_results = []
    for process_result in process_results:
        all_results.extend(process_result[0])
        all_query_results.extend(process_result[1])
    results = pd.DataFrame(all_results)
    query_results = pd.DataFrame(all_query_results)

    query_results.to_csv(os.path.join(benchmark_configurations_folder, "query_results.csv"), index=False)

    dictionary_size = workload['all_dictionary_size']
    dictionary_runtime = workload['all_dictionary_runtime']
    results_with_static_dictionary = pd.concat([results, pd.DataFrame({"MODEL": ["Static"], "BUDGET": [dictionary_size], "SIZE_IN_BYTES": [dictionary_size], "CUMULATIVE_RUNTIME_MS": [dictionary_runtime]})])

    # create static dictonary configuration
    dict_configuration = np.zeros((workload['table_count'], workload['max_row_clusters'], workload['max_column_count']), dtype=np.int32)
    encoding_selection_helpers.write_configuration_to_json(benchmark_configurations_folder, "Static", int(dictionary_size),
                                                              dict_configuration, float(dictionary_runtime), workload, model_to_use)

    # plotting in R
    results_with_static_dictionary.to_csv(os.path.join(benchmark_configurations_folder, "model_results.csv"), index=False)

    full_results = pd.read_csv(os.path.join(benchmark_configurations_folder, "model_results.csv"))
    ids_to_drop = []

    for row in full_results.itertuples():
      if row.MODEL.startswith("GreedyAlpha"):
        alternative_name = row.MODEL.replace("GreedyAlpha", "GreedyBackwardsAlpha")
        alternative = full_results.query("MODEL == @alternative_name and BUDGET == @row.BUDGET")
        if len(alternative) > 0:
          alternative_runtime = alternative.CUMULATIVE_RUNTIME_MS.values[0]
          if alternative_runtime <= row.CUMULATIVE_RUNTIME_MS:
            ids_to_drop.append(row[0])

      if row.MODEL.startswith("GreedyBackwardsAlpha"):
        alternative_name = row.MODEL.replace("GreedyBackwardsAlpha", "GreedyAlpha")
        alternative = full_results.query("MODEL == @alternative_name and BUDGET == @row.BUDGET")
        if len(alternative) > 0:
          alternative_runtime = alternative.CUMULATIVE_RUNTIME_MS.values[0]
          if alternative_runtime < row.CUMULATIVE_RUNTIME_MS:
            ids_to_drop.append(row[0])

    full_results = full_results.drop(index=ids_to_drop)
    full_results.to_csv(os.path.join(benchmark_configurations_folder, "model_results__greedy_filtered.csv"), index=False)

