#!/usr/bin/python3
# Thanks to Markus Dreseler, who initially built this script.

import argparse
import glob
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time

from datetime import datetime
from pathlib import Path

import atexit
import sys
import duckdb
import psycopg2
import pymonetdb

from helpers import static_tpch_queries


# For a fair comparison, we use the same queries as the Umbra demo does.
tpch_queries = static_tpch_queries.queries


job_queries = []
for filename in Path("job_queries").glob("*.sql"):
  if not filename.is_file():
    continue

  with open(str(filename), "r") as sql_file:
    sql_query = sql_file.read()
    job_queries.append([sql_query.strip()]) 


# gather size information
tables = {"JOB":  ["aka_name", "aka_title", "cast_info", "char_name", "comp_cast_type", "company_name", "company_type",
                   "complete_cast", "info_type", "keyword", "kind_type", "link_type", "movie_companies", "movie_info",
                   "movie_info_idx", "movie_keyword", "movie_link", "name", "person_info", "role_type", "title"],
          "TPC-H": ["nation", "region", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]}


parser = argparse.ArgumentParser()
parser.add_argument('dbms', type=str, choices=['monetdb', 'hyrise', 'duckdb', 'umbra'])
parser.add_argument('--time', '-t', type=int, default=300)
parser.add_argument('--port', '-p', type=int, default=5432)
parser.add_argument('--clients', type=int, default=1)
parser.add_argument('--cores', type=int, default=1)
parser.add_argument('--scale_factor', '-s', type=float, default=10.0)
parser.add_argument('--benchmark', '-b', type=str, default="TPC-H")
parser.add_argument('--hyrise_server_path', type=str, default="~/hyrise/cmake-release-build")
parser.add_argument('--single_query_id', type=int)
parser.add_argument('--determine_size_only', action="store_true")
parser.add_argument('--skip_warmup', action="store_true")
args = parser.parse_args()

if args.dbms == 'hyrise':
  hyrise_server_path = Path(args.hyrise_server_path).expanduser().resolve()
  assert (hyrise_server_path / "hyriseServer").exists(), "Please pass valid --hyrise_server_path"

monetdb_scale_factor_string = str(args.scale_factor).replace(".", "_") if float(int(args.scale_factor)) == args.scale_factor else str(int(args.scale_factor))
duckdb_scale_factor_string = int(args.scale_factor) if args.scale_factor >= 1.0 else args.scale_factor

assert (args.single_query_id is None or (args.single_query_id > 0 and args.single_query_id < 23)), "Unexpected query id"
assert (args.dbms != "duckdb" or Path("{}/tpch-dbgen/sf{}/nation.tbl".format(Path.home(), duckdb_scale_factor_string)).exists()), "Expecting TPC-H dbgen data to be present under fixed path."
assert (args.dbms != "monetdb" or Path("{}/monetdb_farm/SF-{}".format(Path.home(), monetdb_scale_factor_string)).exists()), "Expecting MonetDB farm for requested scale factor to be present under fixed path."
assert (args.benchmark != "JOB" or args.dbms in ["hyrise"]), "For now, this script supports the Join Order Benchmark only for Hyrise."
assert (args.clients == 1 or args.time >= 300), "When multiple clients are set, a shuffled run is initiated which should last at least 300s."

if args.dbms == "duckdb" and args.clients > 1:
  cores_per_client = max(1, int(float(args.cores) / float(args.clients)))


duckdb_load_commands = []
duckdb_load_commands.append("""CREATE TABLE nation ( n_nationkey INTEGER not null, n_name CHAR(25) not null, n_regionkey INTEGER not null, n_comment VARCHAR(152), PRIMARY KEY (N_NATIONKEY) );""")
duckdb_load_commands.append("""CREATE TABLE region ( r_regionkey INTEGER not null, r_name CHAR(25) not null, r_comment VARCHAR(152), PRIMARY KEY (R_REGIONKEY) );""")
duckdb_load_commands.append("""CREATE TABLE part ( p_partkey INTEGER not null, p_name VARCHAR(55) not null, p_mfgr CHAR(25) not null, p_brand CHAR(10) not null, p_type VARCHAR(25) not null, p_size INTEGER not null, p_container CHAR(10) not null, p_retailprice DECIMAL(12,2) not null, p_comment VARCHAR(23) not null, PRIMARY KEY (P_PARTKEY) );""")
duckdb_load_commands.append("""CREATE TABLE supplier ( s_suppkey INTEGER not null, s_name CHAR(25) not null, s_address VARCHAR(40) not null, s_nationkey INTEGER not null, s_phone CHAR(15) not null, s_acctbal DECIMAL(12,2) not null, s_comment VARCHAR(101) not null, PRIMARY KEY (S_SUPPKEY) );""")
duckdb_load_commands.append("""CREATE TABLE partsupp ( ps_partkey INTEGER not null, ps_suppkey INTEGER not null, ps_availqty INTEGER not null, ps_supplycost DECIMAL(12,2) not null, ps_comment VARCHAR(199) not null, PRIMARY KEY (PS_PARTKEY,PS_SUPPKEY) );""")
duckdb_load_commands.append("""CREATE TABLE customer ( c_custkey INTEGER not null, c_name VARCHAR(25) not null, c_address VARCHAR(40) not null, c_nationkey INTEGER not null, c_phone CHAR(15) not null, c_acctbal DECIMAL(12,2) not null, c_mktsegment CHAR(10) not null, c_comment VARCHAR(117) not null, PRIMARY KEY (C_CUSTKEY) );""")
duckdb_load_commands.append("""CREATE TABLE orders ( o_orderkey INTEGER not null, o_custkey INTEGER not null, o_orderstatus CHAR(1) not null, o_totalprice DECIMAL(12,2) not null, o_orderdate DATE not null, o_orderpriority CHAR(15) not null, o_clerk  CHAR(15) not null, o_shippriority INTEGER not null, o_comment VARCHAR(79) not null, PRIMARY KEY (O_ORDERKEY) );""")
duckdb_load_commands.append("""CREATE TABLE lineitem ( l_orderkey INTEGER not null, l_partkey INTEGER not null, l_suppkey INTEGER not null, l_linenumber INTEGER not null, l_quantity DECIMAL(12,2) not null, l_extendedprice DECIMAL(12,2) not null, l_discount DECIMAL(12,2) not null, l_tax  DECIMAL(12,2) not null, l_returnflag CHAR(1) not null, l_linestatus CHAR(1) not null, l_shipdate DATE not null, l_commitdate DATE not null, l_receiptdate DATE not null, l_shipinstruct CHAR(25) not null, l_shipmode CHAR(10) not null, l_comment VARCHAR(44) not null, PRIMARY KEY (L_ORDERKEY,L_LINENUMBER) );""")

duckdb_load_commands.append("""COPY nation FROM '{}/tpch-dbgen/sf{}/nation.tbl' ( DELIMITER '|');""".format(Path.home(), duckdb_scale_factor_string))
duckdb_load_commands.append("""COPY region FROM '{}/tpch-dbgen/sf{}/region.tbl' ( DELIMITER '|');""".format(Path.home(), duckdb_scale_factor_string))
duckdb_load_commands.append("""COPY part FROM '{}/tpch-dbgen/sf{}/part.tbl' ( DELIMITER '|');""".format(Path.home(), duckdb_scale_factor_string))
duckdb_load_commands.append("""COPY supplier FROM '{}/tpch-dbgen/sf{}/supplier.tbl' ( DELIMITER '|');""".format(Path.home(), duckdb_scale_factor_string))
duckdb_load_commands.append("""COPY partsupp FROM '{}/tpch-dbgen/sf{}/partsupp.tbl' ( DELIMITER '|');""".format(Path.home(), duckdb_scale_factor_string))
duckdb_load_commands.append("""COPY customer FROM '{}/tpch-dbgen/sf{}/customer.tbl' ( DELIMITER '|');""".format(Path.home(), duckdb_scale_factor_string))
duckdb_load_commands.append("""COPY orders FROM '{}/tpch-dbgen/sf{}/orders.tbl' ( DELIMITER '|');""".format(Path.home(), duckdb_scale_factor_string))
duckdb_load_commands.append("""COPY lineitem FROM '{}/tpch-dbgen/sf{}/lineitem.tbl' ( DELIMITER '|');""".format(Path.home(), duckdb_scale_factor_string))


dbms_process = None
duckdb_connection = None
def cleanup():
  if dbms_process:
    print("Shutting {} down...".format(args.dbms))
    dbms_process.kill()
    time.sleep(10)
atexit.register(cleanup)

print("Starting {}...".format(args.dbms))
if args.dbms == 'monetdb':
  subprocess.Popen(['pkill', '-9', 'mserver5'])
  time.sleep(5)
  cmd = ['numactl', '-C', f"+0-+{args.cores - 1}", 'mserver5', '--dbpath={}/monetdb_farm/SF-{}'.format(Path.home(), monetdb_scale_factor_string),
   '--set', 'monet_vault_key={}/monetdb_farm/SF-{}/.vaultkey'.format(Path.home(), monetdb_scale_factor_string),
         '--set', 'gdk_nr_threads={}'.format(args.cores)]
  if args.clients > 62:
      cmd.extend(["--set", f"max_clients={args.clients + 2}"])
  if args.clients < 33:
      # We have seen strange errors when using the inmemory option and 64 clients (bat file not existing)
      cmd.append("--dbextra=inmemory")
  dbms_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
  while True:
    line = dbms_process.stdout.readline()
    if b'MonetDB/SQL module loaded' in line:
      break
elif args.dbms == 'hyrise':
  dbms_process = subprocess.Popen(['{}/hyriseServer'.format(hyrise_server_path), '-p', str(args.port), '--benchmark_data=TPC-H:{}'.format(str(args.scale_factor))], stdout=subprocess.PIPE)
  time.sleep(5)
  while True:
    line = dbms_process.stdout.readline()
    if b'Server started at' in line:
      break
elif args.dbms == 'umbra':
  parallel_dir = {'PARALLEL': 'off'} if args.cores == 1 else {'PARALLEL': str(args.cores)}
  dbms_process = subprocess.Popen(['numactl', '-C', "+0-+{}".format(args.cores - 1), '{}/umbra/bin/server'.format(Path.home()), '{}/umbra/sf{}/tpch.db'.format(Path.home(), int(args.scale_factor))], stdout=subprocess.DEVNULL, env=parallel_dir)
  print("Waiting 10s for Umbra to start ... ", end="")
  time.sleep(10)
  print("done.")
elif args.dbms == 'duckdb':
  # there is no dbms_process
  Path("db.duckdb").unlink(missing_ok=True)
  duckdb_connection = duckdb.connect(database="db.duckdb", read_only=False)
  cursor = duckdb_connection.cursor()
  # This should be done once, not every time (remainder of previous DuckDB tests).
  print("Generating TPC-H data: ", end="")
  for cmd in duckdb_load_commands:
    cursor.execute(cmd)
    print("{} ".format(cmd.split(" ")[0]), end="", flush=True)
  print("done.")
  cursor.close()
  duckdb_connection.close()


def get_cursor():
  if args.dbms == 'monetdb':
    connection = None
    while connection is None:
      try:
        connection = pymonetdb.connect('SF-{}'.format(monetdb_scale_factor_string), connect_timeout=600)
      except:
        e = sys.exc_info()[0]
        print(e)
        time.sleep(1)
    connection.settimeout(600)
  elif args.dbms == 'hyrise':
    connection = psycopg2.connect("host=localhost port={}".format(args.port))
  elif args.dbms == 'umbra':
    connection = psycopg2.connect(host="/tmp", user="postgres")
  elif args.dbms == 'duckdb':
    connection = duckdb.connect(database="db.duckdb", read_only=True)

  cursor = connection.cursor()
  return (connection, cursor)


def get_aggregated_table_size():
  connection, cursor = get_cursor()
  
  if args.dbms not in ["umbra", "monetdb", "duckdb"]:
    # projecting all tables to load them into memory (DBMS X only reports loaded tables accurately)
    for table in tables[args.benchmark]:
      cursor.execute("SELECT * FROM {}".format(table))
      rows_fetched = 0
      print("{} - Fetching rows of table {}: ".format(datetime.now(), table), flush=True, end="")
      while True:
        rows = cursor.fetchmany(10000)
        if not rows or len(rows) == 0:
          break

        rows_fetched += len(rows)
      print("{:,} rows.".format(rows_fetched), flush=True)

  with open("db_comparison_results/size_{}__SF{}.csv".format(args.dbms, args.scale_factor), "w") as size_file:
    size_file.write("DATABASE_SYSTEM,SCALE_FACTOR,SIZE_IN_BYTES\n")
    cumulative_size = 0
    if args.dbms == "monetdb":
      cursor.execute("select location from storage() where table in ('part', 'partsupp', 'supplier', 'customer', 'orders', 'lineitem', 'nation', 'region');")

      files = cursor.fetchall()
      for file in files:
        monetdb_files = glob.glob("{}/monetdb_farm/SF-{}/bat/{}*".format(Path.home(), monetdb_scale_factor_string, file[0]))
        for monetdb_file in monetdb_files:
          cumulative_size += Path(monetdb_file).stat().st_size
    elif args.dbms == "hyrise" :
      cursor.execute("SELECT SUM(size_in_bytes) FROM meta_segments_accurate;")
      cumulative_size = cursor.fetchone()[0]
    elif args.dbms == "umbra":
      db_files = glob.glob("{}/umbra/sf{}/*.d*".format(Path.home(), int(args.scale_factor)))
      for db_file in db_files:
        # .db (schema) and .data files (relations) are our concern, nothing else (.pages stores indexes)
        if Path(db_file).suffix not in [".db", ".data"]:
          continue
        cumulative_size += Path(db_file).stat().st_size
    elif args.dbms == "duckdb":
      cumulative_size += Path("db.duckdb").stat().st_size

    assert cumulative_size > 0, "Database size of zero probably signals an error."
    size_file.write("{},{},{}\n".format(args.dbms, args.scale_factor, cumulative_size))


def adapt_query(query):
  return query


def loop(thread_id, queries, query_id, start_time, successful_runs, timeout, is_warmup=False):
  connection, cursor = get_cursor()

  if is_warmup:
    if args.skip_warmup:
      return

    for q_id, query in enumerate(queries):
      cursor.execute(adapt_query(query))
      print("({})".format(q_id+1), end="", flush=True)

    cursor.close()
    connection.close()
    return

  while True:
    if query_id == 'shuffled':
      items = queries.copy()
      random.shuffle(items)
    else:
      items = [queries[query_id - 1]]
    item_start_time = time.time()
    for query in items:
      cursor.execute(adapt_query(query))
      cursor.fetchall()
      item_end_time = time.time()

    if (time.time() - start_time < timeout) or len(successful_runs) == 0:
      successful_runs.append((item_end_time - item_start_time) * 1000)
    else:
      break

  if args.dbms != 'duckdb':
    cursor.close()
    connection.close()


if args.determine_size_only:
  get_aggregated_table_size()
  sys.exit()


selected_benchmark_queries = tpch_queries if args.benchmark == "TPC-H" else job_queries

if args.dbms in ['monetdb', 'umbra']:
  print("Warming up database (complete single-threaded TPC-H run) due to initial persistence on disk: ", end="")
  sys.stdout.flush()
  loop(0, selected_benchmark_queries, 'shuffled', time.time(), [], 3600, True)
  print(" done.")
  sys.stdout.flush()

os.makedirs("db_comparison_results", exist_ok=True)

runtimes = {}
benchmark_queries = [args.single_query_id] if args.single_query_id else list(range(1, len(selected_benchmark_queries)+1))
if args.clients > 1:
  benchmark_queries = ["shuffled"]
for query_id in benchmark_queries:
  query_name = '{} {:02}'.format(args.benchmark, query_id) if query_id != 'shuffled' else 'shuffled'
  print('Benchmarking {}...'.format(query_name), end='', flush=True)

  successful_runs = []
  start_time = time.time()

  timeout = args.time

  threads = []
  for thread_id in range (0, args.clients):
    threads.append(threading.Thread(target=loop, args=(thread_id, selected_benchmark_queries, query_id, start_time, successful_runs, timeout)))
    threads[-1].start()

  while True:
    time_left = start_time + timeout - time.time()
    if time_left < 0:
      break
    print('\rBenchmarking {}... {:.0f} seconds left'.format(query_name, time_left), end="", flush=True)
    time.sleep(min(10, time_left))

  while True:
    joined_threads = 0
    for thread_id in range (0, args.clients):
      if not threads[thread_id].is_alive():
        # print(f't{thread_id} finished')
        joined_threads += 1

    if joined_threads == args.clients:
      break
    else:
      print('\rBenchmarking {}... waiting for {} more clients to finish'.format(query_name, args.clients - joined_threads), end="")
      time.sleep(1)

  print('\r' + ' ' * 80, end='')
  print('\r{}\t>>\t avg.: {:10.4f} ms\tmed.: {:10.4f} ms\tmin.: {:10.4f} ms\tmax.: {:10.4f} ms'.format(query_name, sum(successful_runs) / len(successful_runs),
                                                                                                       statistics.median(successful_runs), min(successful_runs),
                                                                                                       max(successful_runs)))

  runtimes[query_name] = successful_runs

result_csv_filename = "db_comparison_results/database_comparison__{}__{}.csv".format(args.benchmark, args.dbms)
result_csv_exists = Path(result_csv_filename).exists()
with open(result_csv_filename, "a" if result_csv_exists else "w") as result_csv:
  if not result_csv_exists:
    result_csv.write("BENCHMARK,DATABASE_SYSTEM,SCALE_FACTOR,CORES,CLIENTS,ITEM_NAME,RUNTIME_MS\n")
  for item_name, runs in runtimes.items():
    for run in runs:
      result_csv.write("{},{},{},{},{},{},{}\n".format(args.benchmark, args.dbms, args.scale_factor, args.cores, args.clients, item_name, run))


