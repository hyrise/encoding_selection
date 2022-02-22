#!/usr/bin/python3

import os
import sys
import argparse

from pathlib import Path

import model_pipeline
import encoding_configuration_selector

parser = argparse.ArgumentParser()
parser.add_argument('--calibration_dir', type=str, required=True)
parser.add_argument('--workload_dir', type=str, required=False)
parser.add_argument('--budget_steps_stretch_factor', type=float, default=1.0,required=False)  # use for faster runs, as budgets steps are larger
parser.add_argument('--use_calibration_as_workload', action='store_true', required=False)
parser.add_argument('--skip_phases', type=str, nargs='+', choices=["load_csv", "prepare", "learn_runtime",
                                                                   "learn_size", "selection"],
                    help="List of phases to skip", default=[])

args = parser.parse_args()

calibration_dir = args.calibration_dir
workload_dir = args.workload_dir

assert Path(calibration_dir).exists(), "Provided calibration directory does not exist."
if workload_dir is not None:
    assert Path(workload_dir).exists(), "Provided workload directory does not exist."
else:
    if "selection" not in args.skip_phases:
        assert args.use_calibration_as_workload == True, "If the calibration directory is also used as the workload "\
                                                         "directory for the actual selection, the "\
                                                         "`--use_calibration_as_workload` flag must be used."
    workload_dir = calibration_dir

print(f"Using calibration run '{Path(calibration_dir).name}' and workload '{Path(workload_dir).name}'")

calibration_result_dir = os.path.join(calibration_dir, "results")

if "load_csv" not in args.skip_phases:
    model_pipeline.load_csv_and_process_dags(calibration_result_dir)
if "prepare" not in args.skip_phases:
    model_pipeline.prepare_learning_data(calibration_result_dir)
if "learn_runtime" not in args.skip_phases:
    model_pipeline.learn_runtime_models(calibration_result_dir)
if "learn_size" not in args.skip_phases:
    model_pipeline.learn_size_models(calibration_result_dir)


if "selection" in args.skip_phases:
    sys.exit()

calibration_dir_models_folder = os.path.join("generated_models", Path(calibration_dir).stem)
dirs = [x for x in Path(os.path.join(workload_dir, "results")).iterdir()]
for dir in sorted(dirs, reverse=True):
    if not dir.is_dir() or "processed_" in str(dir):
        continue

    robustness_constraints = [
                              None,
                              ("QuerySlowdown", {"TPC-H 18": 1.0}),
                              ("QuerySlowdown", {"TPC-H 01": 1.0, "TPC-H 13": 1.0, "TPC-H 21": 1.0}),
                             ]

    print("#######")
    print(f"####### {dir}")
    print("#######")

    workloads_to_skip = []
    workloads_to_skip.append("TPCDS")
    workloads_to_skip.append("JOB")

    skip_workload = False
    for workload_to_skip in workloads_to_skip:
        if workload_to_skip in str(dir):
            skip_workload = True

    if skip_workload:
        print(f"Skipping workloads {workloads_to_skip} for now.")
        continue

    benchmark = dir.stem
    for robustness_constraint in robustness_constraints:
        if robustness_constraint is not None and "TPCH" not in str(dir):
            continue

        if robustness_constraint is not None:
            print(f"### Solving with constraint {robustness_constraint}.")
        encoding_configuration_selector.run_compression_selection_comparison(benchmark, Path(calibration_dir).stem,
                                                                             robustness_constraint,
                                                                             os.path.join(str(dir), "dictionary_fwi"),
                                                                             calibration_dir_models_folder,
                                                                             "XGBoostReg",
                                                                             args.budget_steps_stretch_factor)

