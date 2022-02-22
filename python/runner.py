#!/usr/bin/python3

import argparse
import atexit
import hashlib
import json
import os
import psycopg2
import random
import subprocess
import sys
import threading
import time

from datetime import datetime
from pathlib import Path
from pathlib import Path, PurePath


SCALE_FACTORS = {'TPC-H': 10, 'TPC-DS': 10, 'Join Order Benchmark': 1}
# SCALE_FACTORS = {'TPC-H': 0.1, 'TPC-DS': 1, 'Join Order Benchmark': 1}
BENCHMARKS = [('TPCH', 'TPC-H'), ('TPCDS', 'TPC-DS'), ('JOB', 'Join Order Benchmark')]

PLUGIN_FILETYPE = '.so' if sys.platform == "linux" else '.dylib'

parser = argparse.ArgumentParser()
parser.add_argument('--execute', type=str, choices=["calibration", "evaluation", "timeseries"], required=True)
parser.add_argument('--hyrise_server_path', type=str, default="~/hyrise/cmake-release-build", required=True)
parser.add_argument('--base_benchmark_runs', '-r', type=int, default=2)
parser.add_argument('--port', '-p', type=int, default=5432)
parser.add_argument('--clients', type=int, default=1)  # as soon as we have clients > 1, we perform shuffled runs
# used to run only a single benchmark calibration or tell the evaluation which data set to load
parser.add_argument('--cores', type=int, default=1)
parser.add_argument('--random_encoding_configs_count', type=int, default=20)  # number of randomized configs to generate
parser.add_argument('--single_benchmark', type=str)
parser.add_argument('--single_benchmark_query', type=str)
parser.add_argument('--job_data_path', type=str, default="~/imdb_data")
parser.add_argument('--evaluation_executions', type=int, default=5)
parser.add_argument('--scale_factor', type=float, default=10.0)
parser.add_argument('--configurations_dir', type=str)
parser.add_argument('--results_dir', type=str)
parser.add_argument('--hash_splitting', type=str)  # used for parallel execution on large servers
parser.add_argument('--use_static_tpch_queries', action="store_true")  # use fixed TPC-H queries (taken from Umbra demo) for evaluation
parser.add_argument('--initial_timeseries_config', type=str)
parser.add_argument('timeseries_configs', nargs='*', help="List of configurations to apply")

args = parser.parse_args()
hyrise_server_path = Path(args.hyrise_server_path).expanduser().resolve()
hyrise_server_process = None

assert Path(hyrise_server_path.joinpath('resources/benchmark/tpcds/tpcds-result-reproduction/query_qualification')).exists(), \
       "Could not read resource files (needed for TPC-DS). Try building plugin targets explicitely again to create symlinks."

assert args.execute != "evaluation" or (args.configurations_dir is not None and args.results_dir is not None and args.single_benchmark is not None), \
       "When running an evaluation, 'configurations_dir', 'results_dir', and 'single_benchmark' must be passed."
assert args.single_benchmark_query is None or args.single_benchmark is not None, "When 'single_benchmark_query' is requested (mostly used for debugging), \
       'single_benchmark' needs to be set as well."
assert args.hash_splitting is None or (":" in args.hash_splitting and len([arg for arg in args.hash_splitting.split(":") if arg.isdigit()]) == 2)
assert not args.use_static_tpch_queries or args.execute == "evaluation",  "Use static TPC-H queries only for evaluation."
assert not args.use_static_tpch_queries or args.single_benchmark == "TPC-H",  "Static query set is only valid for TPC-H."

# calibration is always executed single-threaded
assert args.execute != "calibration" or args.clients == 1

assert args.execute != "timeseries" or len(args.timeseries_configs) > 0, "Expecting at least one configurations to use for the timeseries evaluation."
assert len(args.timeseries_configs) == 0 or args.initial_timeseries_config is not None, "For the timeseries evaluation, \
       an initial start configuration must be provided."


# A 2 executions LZ4 JOB-run takes 1h. We aim to yield the same runtime per configuration for all benchmarks.
base_query_executions = args.base_benchmark_runs
EXECUTION_COUNTS = {"TPC-H": 3*base_query_executions,
                    "TPC-DS": 2*base_query_executions,
                    "Join Order Benchmark": base_query_executions}
if args.single_benchmark_query is not None:
  EXECUTION_COUNTS = {key: base_query_executions for key in EXECUTION_COUNTS.keys()}


# clear log
log_file = open("hyrise_server.log", "w")
log_file.flush()
log_file.close()

calibration_directory = os.path.join("calibration", str(datetime.today().strftime("%Y-%m-%dT%H%M%S")))
Path(calibration_directory).mkdir(parents=True, exist_ok=True)

benchmarks = BENCHMARKS
if args.single_benchmark: 
  benchmarks = [b for b in BENCHMARKS if args.single_benchmark in b]
  assert(len(benchmarks) == 1)

calibration_summary = {}
calibration_summary['benchmarks'] = {}
for benchmark_short, benchmark_long in benchmarks:
  calibration_summary['benchmarks'][benchmark_long] = {}
  calibration_summary['benchmarks'][benchmark_long]['short_name'] = benchmark_short
  calibration_summary['benchmarks'][benchmark_long]['scale_factor'] = SCALE_FACTORS[benchmark_long] if not args.scale_factor else args.scale_factor
  calibration_summary['benchmarks'][benchmark_long]['runs'] = EXECUTION_COUNTS[benchmark_long]
calibration_summary['start'] = str(datetime.today().strftime('%Y-%m-%dT%H:%M:%S.%f'))

summary_file_name = os.path.join(calibration_directory, 'summary.json')
with open(summary_file_name, 'w') as summary_file:
    json.dump(calibration_summary, summary_file, indent=2)


def cleanup():
  global hyrise_server_process
  if hyrise_server_process:
    print("Shutting down...")
    hyrise_server_process.kill()
    while hyrise_server_process.poll() is None:
      time.sleep(1)
atexit.register(cleanup)


def get_column_names_from_cursor(cursor):
  names = {}
  for item_id, item in enumerate(cursor.description):
    names[item.name] = item_id
  return names


def get_header_only_csv_file(path, cursor, file_name, table):
  file = open(os.path.join(path, '{}.csv'.format(file_name)), 'w')

  cursor.execute("SELECT * FROM {};".format(table))
  names = []
  for item in cursor.description:
    names.append(item.name.upper())
  file.write(",".join(names) + "\n")

  return file


def write_operator_data_to_csv_file(table_name, cursor, file):
  cursor.execute("SELECT * FROM {};".format(table_name))
  type_codes = [entry.type_code for entry in cursor.description]
  for row in cursor.fetchall():
    output_fields = []
    for item, type_code in zip(row, type_codes):
      if item is None:
        output_fields.append("NULL")
        continue
      item_output = str(item)
      if type_code in [8,9,10,11,25,29,30,33,35,36,37,51,52,53]:
        # see https://github.com/SAP/PyHDB/blob/master/pyhdb/protocol/constants/type_codes.py
        # but a few added based on try and error
        item_output = '"' + item_output + '"'
      output_fields.append(item_output)
    file.write(','.join(output_fields) + '\n')


# This is most certainly not the best method or in any way efficient, but due to a lack of passing
# "customizable commands" to Hyrise, this is an easy and fast workaround/hack.
def send_command(cursor, command):
  cursor.execute("UPDATE meta_settings SET value='{}' WHERE name='Plugin::Executor::Command';".format(command))
  cursor.execute("SELECT * FROM meta_command_executor;")

  result = cursor.fetchall()
  assert len(result) == 1
  if result[0][0] != "Command executed successfully.":
    sys.exit("Error sending command: {}".format(command))


def load_join_order_benchmark_data(cursor):
  def load_table(name, bin_file, data_path):
    try:
      cursor.execute("COPY {} from '{}/{}';".format(name, data_path, bin_file))
    except Exception as e:  
      sys.exit("Error when loading Join Order benchmark table '{}' ({})".format(name, data_path))
    print(".", end="", flush=True)


  data_path = str(Path(args.job_data_path).expanduser().resolve())
  print('{} - Loading "Join Order Benchmark" data '.format(datetime.now()), end="", flush=True)
  load_table("title", "title.bin", data_path)
  load_table("kind_type", "kind_type.bin", data_path)
  load_table("role_type", "role_type.bin", data_path)
  load_table("link_type", "link_type.bin", data_path)
  load_table("keyword", "keyword.bin", data_path)
  load_table("company_type", "company_type.bin", data_path)
  load_table("movie_link", "movie_link.bin", data_path)
  load_table("complete_cast", "complete_cast.bin", data_path)
  load_table("person_info", "person_info.bin", data_path)
  load_table("comp_cast_type", "comp_cast_type.bin", data_path)
  load_table("aka_title", "aka_title.bin", data_path)
  load_table("movie_info", "movie_info.bin", data_path)
  load_table("company_name", "company_name.bin", data_path)
  load_table("movie_keyword", "movie_keyword.bin", data_path)
  load_table("movie_companies", "movie_companies.bin", data_path)
  load_table("cast_info", "cast_info.bin", data_path)
  load_table("info_type", "info_type.bin", data_path)
  load_table("aka_name", "aka_name.bin", data_path)
  load_table("movie_info_idx", "movie_info_idx.bin", data_path)
  load_table("char_name", "char_name.bin", data_path)
  load_table("name", "name.bin", data_path)
  print(" done", flush=True)


def load_plugins(cursor):
  cursor.execute("""INSERT INTO meta_plugins(name)
                    VALUES ('{}/libWorkloadHandlerPlugin{}');""".format(str(hyrise_server_path), PLUGIN_FILETYPE))
  cursor.execute("""INSERT INTO meta_plugins(name)
                    VALUES ('{}/libWorkloadStatisticsPlugin{}');""".format(str(hyrise_server_path), PLUGIN_FILETYPE))
  cursor.execute("""INSERT INTO meta_plugins(name)
                    VALUES ('{}/libCommandExecutorPlugin{}');""".format(str(hyrise_server_path), PLUGIN_FILETYPE))
  cursor.execute("""INSERT INTO meta_plugins(name)
                    VALUES ('{}/libDataCharacteristicsPlugin{}');""".format(str(hyrise_server_path), PLUGIN_FILETYPE))


def fetch_benchmark_queries(cursor, benchmark_name):
  benchmark_queries = {}
  benchmark_item_filter = ""
  if args.single_benchmark_query is not None:
    benchmark_item_filter = f" and item_name = '{args.single_benchmark_query}'"
  cursor.execute("select * from meta_benchmark_items where benchmark_name = '{}'{};".format(benchmark_name,
                                                                                            benchmark_item_filter))
  col_names = get_column_names_from_cursor(cursor)
  queries = cursor.fetchall()
  assert len(queries) > 0, "Fetched empty set of benchmark queries."
  for row in queries:
    item_name = row[col_names['item_name']]
    if item_name not in benchmark_queries:
      benchmark_queries[item_name] = []

    benchmark_queries[item_name].append(row[col_names["sql_statement_string"]])
  return benchmark_queries

# Writes compression configurations as JSON to disk and return list of paths.
# The JSON files are written in a way to be directly applied by the ChunkEncoder.
def get_calibration_encoding_configs(cursor, folder_suffix):

  def get_encoding_type(encoding_str):
    compr_mapping = {"fwi": "Fixed-width integer",
                     "bp": "Bit-packing"}
    ret = {}
    if "_" in encoding_str:
      ret["SegmentEncodingType"] = encoding_str[:encoding_str.find("_")]
      ret["VectorEncodingType"] = compr_mapping[encoding_str[encoding_str.find("_")+1:]]
      return ret

    ret["SegmentEncodingType"] = encoding_str
    return ret

  supports = {
    "Dictionary": ["int", "float", "string"],
    "Unencoded": ["int", "float", "string"],
    "LZ4": ["int", "float", "string"],
    "RunLength": ["int", "float", "string"],
    "FixedStringDictionary": ["string"],
    "FSST": ["string"],
    "FrameOfReference": ["int"]
  }

  config_definitions = {
    "lz4": {"int": "LZ4",
            "float": "LZ4",
            "string": "LZ4"},
    "dictionary_fwi": {"int": "Dictionary_fwi",
                        "float": "Dictionary_fwi",
                        "string": "Dictionary_fwi"},
    "dictionary_bp": {"int": "Dictionary_bp",
                             "float": "Dictionary_bp",
                             "string": "Dictionary_bp"},
    "unencoded": {"int": "Unencoded",
                  "float": "Unencoded",
                  "string": "Unencoded"},
    "run_length": {"int": "RunLength",
                   "float": "RunLength",
                   "string": "RunLength"},
    "fixed_string_fwi__and__for_fwi": {"int": "FrameOfReference_fwi",
                                         "float": "Dictionary_fwi",
                                         "string": "FixedStringDictionary_fwi"},
    "fixed_string_bp__and__for_bp": {"int": "FrameOfReference_bp",
                                                   "float": "Dictionary_fwi",
                                                   "string": "FixedStringDictionary_bp"},
    "fsst__and__for_fwi": {"int": "FrameOfReference_fwi",
                           "float": "Dictionary_fwi",
                           "string": "FSST"}
  }

  output_directory = os.path.join(calibration_directory, "configurations", folder_suffix)
  Path(output_directory).mkdir(parents=True, exist_ok=True)
  cursor.execute("""select s.table_name as table_name,
                           s.column_name as column_name,
                           column_data_type,
                           chunk_id,
                           column_id
                    from meta_columns c
                    join meta_segments s
                      on c.table_name=s.table_name and c.column_name=s.column_name;""")
  col_names = get_column_names_from_cursor(cursor)
  segment_meta_data = cursor.fetchall()

  file_paths = []
  for config_name, config_definition in config_definitions.items():
    json_file = {}
    json_file['configuration'] = {}
    configuration = json_file['configuration']
    for row in segment_meta_data:
      table_name = row[col_names["table_name"]]
      chunk_id = int(row[col_names["chunk_id"]])
      column_id = int(row[col_names["column_id"]])
      data_type = row[col_names["column_data_type"]]

      if table_name not in configuration:
        configuration[table_name] = {}
      if chunk_id not in configuration[table_name]:
        configuration[table_name][chunk_id] = {}

      encoding_type = get_encoding_type(config_definition[data_type])
      configuration[table_name][chunk_id][column_id] = encoding_type

    file_path = os.path.join(output_directory, "{}.json".format(config_name))
    with open(file_path, "w") as json_config_file:
      json.dump(json_file, json_config_file, indent=2)
    file_paths.append(file_path)

  # Do not simply chose an encoding randomly for a given column as certain encodings (e.g., Dictionary or LZ4) support
  # more data types and are thus more often chosen. Try to balance encoding types. Example: since FoR only supports
  # integers (let's assume column types are uniformly distributed, thus a third of all columns are integer columns), we
  # add FoR three times as often to the list of randomly chosen encodings than dictionary.
  cursor.execute("""SELECT column_data_type, COUNT(*)
                    FROM meta_segments
                    GROUP BY column_data_type""")
  agg_data_types = cursor.fetchall()

  data_type_occurences = {data_type: 0 for data_type in ['int', 'long', 'float', 'double', 'string']}
  for row in agg_data_types:
    data_type_occurences[row[0]] = row[1]

  all_encodings_str = set([enc_str for conf_name, options in config_definitions.items() for _, enc_str in options.items()])
  possible_encoding_usages = {}
  for encoding_str in all_encodings_str:
    usage_count = 0
    encoding = get_encoding_type(encoding_str)
    for supported_data_type in supports[encoding['SegmentEncodingType']]:
      usage_count += data_type_occurences[supported_data_type]
    possible_encoding_usages[encoding_str] = usage_count
  max_usage_count = max(possible_encoding_usages.values())
  encoding_list_to_chose_from = []
  for k, v in possible_encoding_usages.items():
    factor = 10 * (float(max_usage_count) / v)
    encoding_list_to_chose_from.extend(int(factor) * [k])

  for random_config_id in range(args.random_encoding_configs_count):
    assigned = {}
    json_file = {}
    json_file['configuration'] = {}
    configuration = json_file['configuration']
    for row in segment_meta_data:
      table_name = row[col_names["table_name"]]
      chunk_id = int(row[col_names["chunk_id"]])
      column_id = int(row[col_names["column_id"]])
      data_type = row[col_names["column_data_type"]]

      if (table_name, column_id) in assigned:
        encoding_type = assigned[(table_name, column_id)]
      else:
        encoding_type = None
        while True:
          encoding = get_encoding_type(random.choice(encoding_list_to_chose_from))
          if data_type in supports[encoding['SegmentEncodingType']]:
            encoding_type = encoding
            break

      if table_name not in configuration:
        configuration[table_name] = {}
      if chunk_id not in configuration[table_name]:
        configuration[table_name][chunk_id] = {}

      configuration[table_name][chunk_id][column_id] = encoding_type
      assigned[(table_name, column_id)] = encoding_type

    file_path = os.path.join(output_directory, "random__{:02d}.json".format(random_config_id))
    with open(file_path, "w") as json_config_file:
      json.dump(json_file, json_config_file, indent=2)
    file_paths.append(file_path)

  return file_paths


def run_queries(thread_id, results, benchmark_queries, execution_runs, run_event, is_calibration, output_folder):
  is_single_client = (args.clients == 1)
  connection = psycopg2.connect("host=localhost port={}".format(args.port))
  connection.autocommit = True
  cursor = connection.cursor()

  if is_calibration:
    assert is_single_client
    # create CSV output files
    aggregates_csv = get_header_only_csv_file(output_folder, cursor, "aggregates", "meta_plan_cache_aggregates")
    joins_csv = get_header_only_csv_file(output_folder, cursor, "joins", "meta_plan_cache_joins")
    projections_csv = get_header_only_csv_file(output_folder, cursor, "projections", "meta_plan_cache_projections")
    table_scans_csv = get_header_only_csv_file(output_folder, cursor, "table_scans", "meta_plan_cache_table_scans")
    get_tables_csv = get_header_only_csv_file(output_folder, cursor, "get_tables", "meta_plan_cache_get_tables")
    misc_operators_csv = get_header_only_csv_file(output_folder, cursor, "misc_operators", "meta_plan_cache_misc_operators")

    plan_cache_csv = open(os.path.join(output_folder, "plan_cache.csv"), "w")
    plan_cache_csv.write("QUERY_HASH,QUERY_STATEMENT_HASH,ITEM_NAME,EXECUTION_COUNT,AVG_RUNTIME_MS,EXECUTION_TIMESTAP,QUERY_STRING\n")

  queries = [(k, v) for k, v in benchmark_queries.items()]
  if not is_single_client:
    random.shuffle(queries)

  tmp_results = []
  run = 0
  while (run < execution_runs + 1 and not run_event.is_set()):  # + 1 for warmup execution
    if is_calibration:
      print("{}: run {} ".format(datetime.now(), run), flush=True)

    all_queries_start_time = time.time()
    for item_name, item_instances in queries:
      item_id = run % len(item_instances)
      evaluation_item = item_instances[item_id]

      if not args.use_static_tpch_queries and item_name == "TPC-H 15":
        assert evaluation_item.startswith("create view ")
        revenue_str = evaluation_item[evaluation_item.find(" revenue")+1:evaluation_item.find(" (")]
        evaluation_item = evaluation_item.replace(revenue_str, "{}TID{}".format(revenue_str, thread_id))

      item_start_time = time.time()
      cursor.execute(evaluation_item)
      cursor.fetchall()
      item_end_time = time.time()
      runtime = (item_end_time - item_start_time) * 1000.0

      if run > 0:
        tmp_results.append([item_name, item_end_time, runtime, str(datetime.today().strftime('%Y-%m-%dT%H:%M:%S.%f'))])

        if is_calibration:
          # gather PQP export information
          write_operator_data_to_csv_file('meta_plan_cache_aggregates', cursor, aggregates_csv)
          write_operator_data_to_csv_file('meta_plan_cache_joins', cursor, joins_csv)
          write_operator_data_to_csv_file('meta_plan_cache_projections', cursor, projections_csv)
          write_operator_data_to_csv_file('meta_plan_cache_table_scans', cursor, table_scans_csv)
          write_operator_data_to_csv_file('meta_plan_cache_get_tables', cursor, get_tables_csv)
          write_operator_data_to_csv_file('meta_plan_cache_misc_operators', cursor, misc_operators_csv)

          # Admittedly a little bit ugly, but to obtain the query identifier (std::hash + hex), we access Hyrise again.
          # There is probably a simpler way to obtain a good query identifier on both platforms.
          cursor.execute("SELECT TOP 1 query_hash, query_statement_hash FROM meta_plan_cache_table_scans;")
          result = cursor.fetchone()
          query_hash = result[0]
          query_statement_hash = result[1]
          plan_cache_csv.write("""{},{},"{}",1,{:.9f},{},"{}"\n""".format(query_hash, query_statement_hash, item_name, runtime,
                                                                    str(datetime.today().strftime('%Y-%m-%dT%H:%M:%S.%f')),
                                                                    " ".join(evaluation_item.replace("\n", " ").replace('"', '""').split())))
      
      if is_calibration:
        # here to ensure that warm up run is cleared
        send_command(cursor, "DROP PQP PLANCACHE")
    
    all_queries_end_time = time.time()
    all_queries_runtime = (all_queries_end_time - all_queries_start_time) * 1000.0
    tmp_results.append(["All queries", all_queries_end_time, all_queries_runtime, str(datetime.today().strftime('%Y-%m-%dT%H:%M:%S.%f'))])

    run += 1

  if not is_calibration:
    results[thread_id] = tmp_results

  if is_calibration:
    aggregates_csv.close()
    joins_csv.close()
    projections_csv.close()
    table_scans_csv.close()
    get_tables_csv.close()
    misc_operators_csv.close()

    plan_cache_csv.close()

    table_meta_data = get_header_only_csv_file(output_folder, cursor, "meta_data_table", "meta_tables")
    write_operator_data_to_csv_file("meta_tables", cursor, table_meta_data)
    table_meta_data.close()

    chunk_meta_data = get_header_only_csv_file(output_folder, cursor, "meta_data_chunk", "meta_chunks")
    write_operator_data_to_csv_file("meta_chunks", cursor, chunk_meta_data)
    chunk_meta_data.close()

    column_meta_data = get_header_only_csv_file(output_folder, cursor, "meta_data_column", "meta_columns")
    write_operator_data_to_csv_file("meta_columns", cursor, column_meta_data)
    column_meta_data.close()

    segment_meta_data = get_header_only_csv_file(output_folder, cursor, "meta_data_segment", "meta_segments_accurate")
    write_operator_data_to_csv_file("meta_segments_accurate", cursor, segment_meta_data)
    segment_meta_data.close()

  return True


def run_compression_tasks(job_count, configurations, results, compression_tasks_event):
  connection = psycopg2.connect("host=localhost port={}".format(args.port))
  connection.autocommit = True
  cursor = connection.cursor()

  tpch_scale_factor = SCALE_FACTORS["TPC-H"] if not args.scale_factor else args.scale_factor
  time.sleep(max(60, 60 * tpch_scale_factor))

  for task_count in [1, 20]:
    for i, configuration in enumerate(configurations):
      print("{} - Start compression config #{} with {} tasks ...".format(datetime.now(), i, task_count))
      with open(configuration, "r") as json_config_file:
        json_configuration = json.load(json_config_file)
        budget = float(json_configuration["context"]["budget_in_bytes"]) / 1000 / 1000 / 1000
        start_time = time.time()
        send_command(cursor, 'APPLY COMPRESSION CONFIGURATION {} Scheduler {}'.format(configuration, task_count))
        results.append(["Applying configuration #{} ({:.1f} GB budget, {} jobs)".format(i, budget, task_count),
                        start_time, time.time()])
        print("{} - Finished compression config #{} with {} jobs.".format(datetime.now(), i, task_count))

      time.sleep(max(60, 60 * tpch_scale_factor))

  compression_tasks_event.set()


def start_hyrise_server(benchmark):
  benchmark_long = benchmark[1]

  global hyrise_server_process
  print("Starting Hyrise server (not NUMA-bound, loading data for {}) ... ".format(benchmark_long))

  scale_factor = SCALE_FACTORS[benchmark_long] if not args.scale_factor else args.scale_factor

  benchmark_data_string = ''
  if benchmark_long != 'Join Order Benchmark':
    benchmark_data_string = '--benchmark_data={}:{}'.format(benchmark_long, str(scale_factor))
  hyrise_server_process = subprocess.Popen(['{}/hyriseServer'.format(str(hyrise_server_path)), '-p', str(args.port),
                                            benchmark_data_string], cwd=hyrise_server_path, stdout=subprocess.PIPE,
                                            bufsize=0, universal_newlines=True)

  server_has_started_condition = threading.Event()
  def log_and_check_server_start(pipe):
    log_file = open("hyrise_server.log", "a", buffering=1)
    for line in iter(pipe.readline, ""):
      if line:
        if 'Server started at' in line:
          server_has_started_condition.set()
        log_file.write('{}:\t{}'.format(datetime.now(), line))       
        sys.stdout.flush()
    pipe.close()
    log_file.close()

  logger_thread = threading.Thread(target=log_and_check_server_start,
                                   args=(hyrise_server_process.stdout,  ))
  logger_thread.daemon = True
  logger_thread.start()

  print("Waiting for Hyrise to start: ", end="")
  server_start_time = time.time()
  while not server_has_started_condition.wait(timeout=5):
    print(".", end="", flush=True)
    if time.time() - min(1200, 120 * scale_factor) > server_start_time:
      print("Error: time out during server start")
      break
  print("done.")


def run_calibration():
  run_calibration_event = threading.Event()  # used for shuffled runs; not used here

  for benchmark in benchmarks:
    benchmark_short = benchmark[0]
    benchmark_long = benchmark[1]
    calibration_summary['benchmarks'][benchmark_long]['start'] = str(datetime.today().strftime('%Y-%m-%dT%H:%M:%S.%f'))
    with open(summary_file_name, 'w') as summary_file:
      json.dump(calibration_summary, summary_file, indent=2)

    benchmark_output_directory = os.path.join(calibration_directory, "results", benchmark_short)
    Path(benchmark_output_directory).mkdir(parents=True, exist_ok=True)

    start_hyrise_server(benchmark)

    # load plugins
    connection = psycopg2.connect("host=localhost port={}".format(args.port))
    connection.autocommit = True
    cursor = connection.cursor()

    if benchmark_long == 'Join Order Benchmark':
      load_join_order_benchmark_data(cursor)

    load_plugins(cursor)

    send_command(cursor, "DROP PQP PLANCACHE")  # to get rid of imports
    send_command(cursor, "SET SERVER CORES 0")

    print("{} - Obtaining data characteristics.".format(datetime.now()), flush=True)
    data_characteristics = get_header_only_csv_file(benchmark_output_directory, cursor,
                                                    'data_characteristics', 'meta_data_characteristics')
    write_operator_data_to_csv_file('meta_data_characteristics', cursor, data_characteristics)
    data_characteristics.close()
    print("{} - Obtaining data characteristics: done.".format(datetime.now()), flush=True)

    benchmark_queries = fetch_benchmark_queries(cursor, benchmark_long)

    for encoding_config_path in get_calibration_encoding_configs(cursor, benchmark_short):
      send_command(cursor, "SET SERVER CORES 0")
      full_path = Path(encoding_config_path).expanduser().resolve()
      print("{} - Applying encoding configuration {}".format(datetime.now(), encoding_config_path), flush=True)
      send_command(cursor, 'APPLY COMPRESSION CONFIGURATION {} Scheduler 0'.format(str(full_path)))
      print("{} - Applying encoding configuration: done.".format(datetime.now()), flush=True)

      encoding_config_output_folder = Path(encoding_config_path).stem
      output_folder = "./{}/{}".format(benchmark_output_directory, encoding_config_output_folder)
      Path(output_folder).mkdir(parents=True, exist_ok=True)

      send_command(cursor, "SET SERVER CORES 1")
      run_queries(0, [], benchmark_queries, EXECUTION_COUNTS[benchmark_long],
                  run_calibration_event, True, output_folder)

    cursor.close()
    connection.close()

    calibration_summary['benchmarks'][benchmark_long]['end'] = str(datetime.today().strftime('%Y-%m-%dT%H:%M:%S.%f'))
    with open(summary_file_name, 'w') as summary_file:
      json.dump(calibration_summary, summary_file, indent=2)

    print("{} - Finished running {}. Shutting down Hyrise ...".format(datetime.now(), benchmark))
    hyrise_server_process.kill()
    while hyrise_server_process.poll() is None:
      time.sleep(1)

  calibration_summary['end'] = str(datetime.today().strftime('%Y-%m-%dT%H:%M:%S.%f'))
  with open(summary_file_name, 'w') as summary_file:
    json.dump(calibration_summary, summary_file, indent=2)


def run_evaluation(benchmark, evaluation_configurations_path):
  start_hyrise_server(benchmark)

  connection = psycopg2.connect("host=localhost port={}".format(args.port))
  connection.autocommit = True
  cursor = connection.cursor()

  scale_factor = SCALE_FACTORS[benchmark[1]] if not args.scale_factor else args.scale_factor

  unsorted_configurations = list(evaluation_configurations_path.rglob('*.json'))
  assert len(unsorted_configurations) > 0, "No configurations found."
  # We create a sorted list of configurations: first the static ones to ensure we have them covered first and second
  # the remainders sorted by the file name to group selection approaches. With that, we also sort by memory budgets to
  # minimize the configuration changes to apply).
  configurations = [c for c in unsorted_configurations if "Static" in str(c)]
  configurations.extend(sorted([c for c in unsorted_configurations if "Static" not in str(c)]))

  if benchmark[1] == 'Join Order Benchmark':
    load_join_order_benchmark_data(cursor)

  load_plugins(cursor)
  send_command(cursor, "SET SERVER CORES {}".format(args.cores))

  # We use these hash splits to allow multiple concurrent evaluation runners (we run them on a large NUMA system and
  # pin each process to a single NUMA node).
  hash_count = 1
  hash_split = 0
  hash_split_suffix = ""
  if args.hash_splitting is not None:
    hash_count = int(args.hash_splitting.split(":")[0])
    hash_split = int(args.hash_splitting.split(":")[1])
    hash_split_suffix = "__{}".format(hash_split)

  if args.use_static_tpch_queries:
    # Usually, we use the large query set. But to compare against MonetDB, Umbra etc. we use the same query set (i.e.,
    # the one that the Umbra demo uses).
    from helpers import static_tpch_queries
    benchmark_queries = static_tpch_queries.queries
  else:
    benchmark_queries = fetch_benchmark_queries(cursor, benchmark[1])

  evaluation_results_directory = Path(args.results_dir).expanduser().resolve()
  evaluation_results_directory.mkdir(parents=True, exist_ok=True)

  with open(os.path.join(evaluation_results_directory, "runtimes__{}_clients_and_{}_cores{}.csv".format(args.clients, args.cores, hash_split_suffix)), "w") as runtimes_csv:
    with open(os.path.join(evaluation_results_directory, "sizes__{}_clients_and_{}_cores{}.csv".format(args.clients, args.cores, hash_split_suffix)), "w") as sizes_csv:
      runtimes_csv.write("MODEL,BUDGET,ITEM_NAME,RUNTIME_MS,EXECUTION_TIMESTAMP\n")
      sizes_csv.write("MODEL,BUDGET,SIZE_IN_BYTES,JSON_CONFIGURATION_FILENAME\n")

      for encoding_config_path in configurations:
        # check if hash splitting is requested, if so, only work on file that meet requirement
        if args.hash_splitting:
          file_name_hash_hex = hashlib.md5(str(encoding_config_path).encode()).hexdigest()
          file_name_hash = int(file_name_hash_hex, 16)
          if (file_name_hash % hash_count) != hash_split:
            continue

        full_path = Path(encoding_config_path).expanduser().resolve()
        configuration_json = open(full_path, "r")
        configuration = json.load(configuration_json)
        encoding_config_name = Path(encoding_config_path).stem
        model = configuration["context"]["selection_model"]
        budget = configuration["context"]["budget_in_bytes"]
        runtime = configuration["context"]["runtime"]

        # Manual handling of greedy heuristic. Only evalute the greedy forwards/backwards variant with the smaller runtime
        if model.startswith("GreedyAlpha"):
          alternative_path = str(full_path).replace("GreedyAlpha", "GreedyBackwardsAlpha")
          if Path(alternative_path).exists():
            configuration2_json = open(alternative_path, "r")
            configuration2 = json.load(configuration2_json)
            runtime2 = configuration2["context"]["runtime"]

            if runtime2 <= runtime:
              print("Backwards alternative is faster, skipping forwards variant.")
              continue

        if model.startswith("GreedyBackwardsAlpha"):
          alternative_path = str(full_path).replace("GreedyBackwardsAlpha", "GreedyAlpha")
          if Path(alternative_path).exists():
            configuration2_json = open(alternative_path, "r")
            configuration2 = json.load(configuration2_json)
            runtime2 = configuration2["context"]["runtime"]

            if runtime2 < runtime:
              print("Forwards alternative is faster, skipping backwards variant.")
              continue
        
        hash_info = "[#{}] ".format(hash_split) if args.hash_splitting else ""
        encoding_start = time.time()
        print("{} - {}Evaluating with encoding configuration {} ... ".format(datetime.now(), hash_info, encoding_config_path), end="")
        send_command(cursor, 'APPLY COMPRESSION CONFIGURATION {} Scheduler 0'.format(str(full_path)))
        print("done ({:.1f} s).".format(time.time() - encoding_start), flush=True)

        run_evaluation_event = threading.Event()

        query_runtimes = [None] * args.clients
        threads = []
        loop_executions = float("inf") if args.clients > 1 else args.evaluation_executions
        for thread_id in range (0, args.clients):
          threads.append(threading.Thread(target=run_queries, args=(thread_id, query_runtimes,
                                                                    benchmark_queries, loop_executions,
                                                                    run_evaluation_event, False, "")))
          threads[-1].start()

        if args.clients > 1:
          time.sleep(6 * scale_factor * 40)
          run_evaluation_event.set()

        for thread in threads:
          thread.join()

        cursor.execute("select sum(size_in_bytes) from meta_segments_accurate;")
        sizes_csv.write("""{},{},{},"{}"\n""".format(model, budget, cursor.fetchone()[0], encoding_config_name))
        sizes_csv.flush()

        for thread_query_runtime in query_runtimes:
            for result in thread_query_runtime:
              runtimes_csv.write("""{},{},"{}",{},{}\n""".format(model, budget, result[0], result[2], result[3]))

  cursor.close()
  connection.close()

  print("Finished running {}. Shutting down Hyrise ...".format(benchmark))
  hyrise_server_process.kill()
  while hyrise_server_process.poll() is None:
    time.sleep(1)


def run_timeseries():
  assert args.clients > 1, "Timeseries evaluation needs more than one client to effectively measure impact of concurrent compression"
  assert Path(args.initial_timeseries_config).exists(), "Passed initial timeseries configuration {} does not exist.".format(config)
  for config in args.timeseries_configs:
    assert Path(config).exists(), "Passed timeseries configuration {} does not exist.".format(config)

  timeseries_configurations = [str(Path(config).expanduser().resolve()) for config in args.timeseries_configs]

  # The first encoding configuration will be already in place. We might want to change that.
  initial_configuration = str(Path(args.initial_timeseries_config).expanduser().resolve())

  core_count = args.cores
  client_count = args.clients

  output_path = "evaluation/timeseries/{}".format(int(time.time()))
  Path(output_path).mkdir(parents=True, exist_ok=True)

  with open(os.path.join(output_path, "metrics.csv"), "w") as metrics_csv:
    metrics_csv.write("SETUP,MEASUREMENT,TIMESTAMP,VALUE\n")
    with open(os.path.join(output_path, "query_runtimes.csv"), "w") as runtimes_csv:
      runtimes_csv.write("SETUP,ITEM_NAME,TIMESTAMP,RUNTIME_S\n")
      with open(os.path.join(output_path, "events.csv"), "w") as events_csv:
        events_csv.write("SETUP,EVENT,TIMESTAMP_START,TIMESTAMP_END\n")

        for job_count in [args.clients]:
          job_name = "CLIENT_COUNT_{}".format(job_count)
          start_hyrise_server(benchmarks[0])

          # load plugins
          connection = psycopg2.connect("host=localhost port={}".format(args.port))
          connection.autocommit = True
          cursor = connection.cursor()

          load_plugins(cursor)
          send_command(cursor, "SET SERVER CORES {}".format(core_count))

          print("{} - Applying initial configuration ... ".format(datetime.now()), end="", flush=True)
          send_command(cursor, 'APPLY COMPRESSION CONFIGURATION {} Scheduler 0'.format(initial_configuration))
          print("done", flush=True)

          # warm up caches for size estimation
          cursor.execute("select sum(estimated_size_in_bytes) from meta_segments;")
          cursor.fetchone()

          benchmark_queries = fetch_benchmark_queries(cursor, "TPC-H")

          results = [None] * client_count
          compressor_results = []
          compression_tasks_event = threading.Event()

          threads = []
          for thread_id in range (0, client_count):
            threads.append(threading.Thread(target=run_queries, args=(thread_id, results,
                                                                      benchmark_queries, float("inf"),
                                                                      compression_tasks_event, False, "")))
            threads[-1].start()

          evaluation_start = time.time()

          compressor_thread = threading.Thread(target=run_compression_tasks, args=(job_count, timeseries_configurations,
                                                                                   compressor_results,
                                                                                   compression_tasks_event))
          compressor_thread.start()

          while not compression_tasks_event.is_set():
            print("{} - Metric loop".format(datetime.now()))

            # Currently, caches are growing pretty much unbound in Hyrise. We clear them regularly to get them out of
            # the calculation.
            send_command(cursor, "DROP PQP PLANCACHE")
            send_command(cursor, "DROP LQP PLANCACHE")

            cursor.execute("SELECT * FROM meta_system_utilization;")
            timestamp = time.time()
            utilization_row = cursor.fetchone()
            metrics_csv.write('"{}","{}",{},{}\n'.format(job_name, "RSS", timestamp, utilization_row[9]))
            metrics_csv.write('"{}","{}",{},{}\n'.format(job_name, "ALLOCATION", timestamp, utilization_row[10]))

            cursor.execute("select sum(estimated_size_in_bytes) from meta_segments;")
            metrics_csv.write('"{}","{}",{},{}\n'.format(job_name, "CUMULATIVE_SEGMENTS_SIZE", time.time(), cursor.fetchone()[0]))

            time.sleep(10)

          compressor_thread.join()
          for thread in threads:
            thread.join()

          cursor.close()
          connection.close()

          print("{} - Finished running with {} cores. Shutting down Hyrise ...".format(datetime.now(), job_count))
          hyrise_server_process.kill()

          for thread_results in results:
            for result in thread_results:
              runtimes_csv.write('"{}","{}",{},{}\n'.format(job_name, result[0], result[1], result[2]))

          for event in compressor_results:
            events_csv.write('"{}","{}",{},{}\n'.format(job_name, event[0], event[1], event[2]))

          while hyrise_server_process.poll() is None:
            time.sleep(1)

          runtimes_csv.flush()
          metrics_csv.flush()
          events_csv.flush()


if __name__ == "__main__":
  if args.execute == "calibration":
    run_calibration()
  elif args.execute == "evaluation":
    assert args.single_benchmark is not None, "Evaluation only works for a given single benchmark as of now."
    assert not args.configurations_dir.startswith("/"), "Please provide a relative path to the 'configurations_dir' parameter"
    assert "~" not in args.configurations_dir, "Please provide a relative path to the 'configurations_dir' parameter"

    evaluation_configurations_path = Path(args.configurations_dir).expanduser().resolve()

    benchmark = benchmarks[0]
    run_evaluation(benchmark, evaluation_configurations_path)
  elif args.execute == "timeseries":
    run_timeseries()
  else:
    sys.exit("Unexpected argument.") 


