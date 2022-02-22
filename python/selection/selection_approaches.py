#!/usr/bin/env python3

import datetime
import numpy as np
import pandas as pd
import pulp
import sys

from sklearn import tree # Abadi-like decision tree

sys.path.append("..")
from helpers import encoding_selection_constants


MIN_CONF_OPTION = min(c.value for c in encoding_selection_constants.EncodingType)
MAX_CONF_OPTION = max(c.value for c in encoding_selection_constants.EncodingType)
COUNT_CONF_OPTIONS = len(encoding_selection_constants.EncodingType)


# We care about runtimes. However, the runtimes stored for now might be for an unsupported encoding.
# Right now, that means we would add the maximum float value (stored in the segment sizes for
# unsupported encodings) to the size. 
# Here, we extract the unsupported encodings from the sizes and set all unsupported alterantives to
# nan both in the runtimes as well as in the sizes.
# This should later change when runtimes and sizes store NULLs for unsupported options.
def prepare_vectors(runtimes, sizes):
        mask = sizes == np.finfo(np.float64).max
        runtimes_adapted = np.where(mask, np.nan, runtimes)
        sizes_adapted = np.where(mask, np.nan, sizes)

        return (runtimes_adapted, sizes_adapted)


class BaseCompressionSelection:
    solver = None
    thread_count = 1
    weight_function = None
    alpha = 1.0

    def __init__(self, workload_):
        self.workload = workload_


    def fit(self, budget):
        pass


    def set_solver(self, solver_):
        self.solver = solver_


    def set_thread_count(self, thread_count_):
        self.thread_count = thread_count_


    def set_weight_function(self, weight_function_):
        self.weight_function = weight_function_


    def set_alpha(self, alpha_):
        self.alpha = alpha_


    def get_full_result(self):
        pass


    def get_final_configuration(self):
        pass


    def print_information(self):
        pass


class StaticBestRatio(BaseCompressionSelection):
    configuration = None
    solving_time = 0.0
    problem_time = 0.0
    size = 0.0
    runtime = 0.0


    def __init__(self, workload_):
        self.workload = workload_



        __table_count = self.workload['table_count']
        __table_dimensions = self.workload['table_dimensions']

        __segment_sizes = self.workload['segment_sizes']
        __segment_sizes_np = self.workload['segment_sizes_np']
        __column_meta_data = self.workload['column_meta_data']
        __runtimes_numpy = self.workload['runtimes_numpy']
        __runtimes_numpy_matrix = self.workload['runtimes_numpy_matrix_greedy']
        __unsupported_data_types_numpy = self.workload['unsupported_data_types_numpy']
        __table_and_column_meta_data = self.workload['table_and_column_meta_data']

        start_problem = datetime.datetime.now()

        heuristic_candidates = []

        # start with smallest configuration
        configuration = np.zeros((__table_count, self.workload['max_row_clusters'],
                                 self.workload['max_column_count']))
        size = 0.0
        runtime = 0.0
        for i in range(__table_count):
            for j in range(__table_dimensions[i][0]): # chunk
                for k in range(__table_dimensions[i][1]): # column
                    # __segment_sizes_np already encounts for supported data types
                    encoding_sizes = __segment_sizes_np[i, j, k]
                    assert(len(encoding_sizes) == COUNT_CONF_OPTIONS)

                    runtimes, sizes = prepare_vectors(__runtimes_numpy_matrix[i, j, k], encoding_sizes)
                    if np.isnan(runtimes).all():
                        # If all encodings are none, the segment is never accessed within the given workload.
                        selected_encoding = np.nanargmin(sizes)
                    else:
                        selected_encoding = np.nanargmin(sizes * runtimes)

                    configuration[i, j, k] = selected_encoding
                    size += encoding_sizes[selected_encoding]

        self.configuration = configuration
        self.size = size


    def fit(self, budget):
        pass


    def get_full_result(self):
        if self.configuration is None:
            return ('Fail', self.runtime, self.size, self.configuration, self.problem_time, self.solving_time)

        result_size = 0.0
        result_runtime = self.workload['all_dictionary_runtime']
        for i in range(self.workload['table_count']):
            for j in range(self.workload['table_dimensions'][i][0]):
                for k in range(self.workload['table_dimensions'][i][1]):
                    result_size += self.workload['segment_sizes_np'][i, j, k, int(self.configuration[i,j,k])]
                    runtime = self.workload['runtimes_numpy_matrix'][i, j, k, int(self.configuration[i,j,k])]
                    if not np.isnan(runtime):
                        result_runtime += runtime

        if not (result_size < self.size * 1.01 and result_size > self.size * 0.99):
            print('static not applicable')

        return ('Success', result_runtime, self.size, self.configuration, self.problem_time, self.solving_time)


    def get_final_configuration(self):
        return self.configuration


PRINT_PLUGIN_INFORMATION = False
class ForwardsGreedyHeuristic(BaseCompressionSelection):
    configuration = None
    solving_time = 0.0
    problem_time = 0.0
    runtime = 0.0

    heuristic_candidates_sorted = None

    initial_configuration = None
    initial_configuration_size = 0.0

    configuration = None
    configuration_size = 0.0


    def __init__(self, workload_, weight_function_, alpha_):
        start_problem = datetime.datetime.now()

        self.workload = workload_
        self.weight_function = weight_function_
        self.alpha = alpha_

        __table_count = self.workload['table_count']
        __table_dimensions = self.workload['table_dimensions']

        __segment_sizes_np = self.workload['segment_sizes_np']
        __column_meta_data = self.workload['column_meta_data']
        __runtimes_numpy_matrix = self.workload['runtimes_numpy_matrix_greedy']
        __unsupported_data_types_numpy = self.workload['unsupported_data_types_numpy']
        __table_and_column_meta_data = self.workload['table_and_column_meta_data']

        start_problem = datetime.datetime.now()
        
        heuristic_candidates = []

        # start with an initial zero'd configuration, then iteratively set the smallest
        # possible encoding and collect information for later greedy heuristic.
        self.initial_configuration = np.zeros((__table_count, self.workload['max_row_clusters'],
                                               self.workload['max_column_count']))
        size = 0.0
        static_config_changes = set()  # debugging stuff for plugin
        for i in range(__table_count):
            table_name = self.workload['table_meta_data'].query('TABLE_ID == @i').iloc[0].name
            for j in range(__table_dimensions[i][0]): # chunk
                for k in range(__table_dimensions[i][1]): # column
                    runtimes, sizes = prepare_vectors(__runtimes_numpy_matrix[i, j, k], __segment_sizes_np[i, j, k])
                    smallest_encoding_id = np.nanargmin(sizes)

                    self.initial_configuration[i, j, k] = smallest_encoding_id
                    size += __segment_sizes_np[i, j, k, smallest_encoding_id]

                    data_type = __table_and_column_meta_data.loc[(i, k), 'DATA_TYPE']
                    for encoding_id in range(COUNT_CONF_OPTIONS):
                        if data_type not in encoding_selection_constants.SUPPORTED_DATA_TYPES[encoding_id]:
                            assert(np.isnan(sizes[encoding_id]))
                            assert(np.isnan(runtimes[encoding_id]))

                    weight_and_alternative = self.weight_function(sizes, runtimes, smallest_encoding_id, self.alpha)
                    if weight_and_alternative:
                        assert __table_and_column_meta_data.loc[(i, k), 'DATA_TYPE'] in encoding_selection_constants.SUPPORTED_DATA_TYPES[weight_and_alternative[1]]
                        heuristic_candidates.append([i, j, k, weight_and_alternative])


        self.initial_configuration_size = size

        # sort all alternative by decreasing weight
        self.heuristic_candidates_sorted = sorted(heuristic_candidates, key=lambda x: x[-1][0])

        end_problem = datetime.datetime.now()
        self.problem_time = (end_problem - start_problem).total_seconds()


    def fit(self, budget):
        start_solving = datetime.datetime.now()
        __segment_sizes_np = self.workload['segment_sizes_np']

        self.configuration = self.initial_configuration.copy()
        self.configuration_size = self.initial_configuration_size

        for i, j, k, alternative in self.heuristic_candidates_sorted:
            alternative_encoding = alternative[1]
            current_size = __segment_sizes_np[i, j, k, int(self.configuration[i, j, k])]
            new_size = __segment_sizes_np[i, j, k, alternative_encoding]
            new_config_size = self.configuration_size + (new_size - current_size)

            if new_config_size > budget:
                # cannot break, because there could be smaller alternatives remaining that fit within the budget
                continue

            self.configuration_size = new_config_size 
            self.configuration[i,j,k] = alternative_encoding

        end_solving = datetime.datetime.now()
        self.solving_time = (end_solving - start_solving).total_seconds()


    def get_full_result(self):
        result_size = 0.0
        result_runtime = self.workload['all_dictionary_runtime']

        for i in range(self.workload['table_count']):
            for j in range(self.workload['table_dimensions'][i][0]):
                for k in range(self.workload['table_dimensions'][i][1]):
                    result_size += self.workload['segment_sizes_np'][i, j, k, int(self.configuration[i,j,k])]
                    runtime = self.workload['runtimes_numpy_matrix'][i, j, k, int(self.configuration[i,j,k])]
                    if not np.isnan(runtime):
                        result_runtime += runtime

        assert(result_size < self.configuration_size * 1.01 and result_size > self.configuration_size * 0.99)

        return ('Success', result_runtime, self.configuration_size, self.configuration, self.problem_time, self.solving_time)


    def get_final_configuration(self):
        return self.configuration


class BackwardsGreedyHeuristic(BaseCompressionSelection):
    configuration = None
    solving_time = 0.0
    problem_time = 0.0
    runtime = 0.0

    heuristic_candidates_sorted = None

    initial_configuration = None
    initial_configuration_size = 0.0

    configuration = None
    configuration_size = 0.0


    def __init__(self, workload_, weight_function_, alpha_):
        self.workload = workload_
        self.weight_function = weight_function_
        self.alpha = alpha_

        __table_count = self.workload['table_count']
        __table_dimensions = self.workload['table_dimensions']

        __segment_sizes_np = self.workload['segment_sizes_np']
        __column_meta_data = self.workload['column_meta_data']
        __runtimes_numpy_matrix = self.workload['runtimes_numpy_matrix_greedy']
        __unsupported_data_types_numpy = self.workload['unsupported_data_types_numpy']
        __table_and_column_meta_data = self.workload['table_and_column_meta_data']

        start_problem = datetime.datetime.now()
        
        heuristic_candidates = []

        # start with an initial zero'd configuration, then iteratively set the fastest
        # possible encoding and collect information for later greedy heuristic.
        self.initial_configuration = np.zeros((__table_count, self.workload['max_row_clusters'],
                                               self.workload['max_column_count']))
        size = 0.0
        static_config_changes = set()  # debugging stuff for plugin
        for i in range(__table_count):
            table_name = self.workload['table_meta_data'].query('TABLE_ID == @i').iloc[0].name
            for j in range(__table_dimensions[i][0]): # chunk
                for k in range(__table_dimensions[i][1]): # column
                    # __runtimes_numpy_matrix should store NA's for unsupported encodings
                    runtimes, sizes = prepare_vectors(__runtimes_numpy_matrix[i, j, k], __segment_sizes_np[i, j, k])
                    if np.isnan(runtimes).all():
                        # If all encodings are none, the segment is never accessed within the given workload.
                        fastest_encoding_id = np.nanargmin(sizes)
                    else:
                        # We care about the smallest runtime. However, this runtime might be for an unsupported encoding.
                        # Right now, that means we would add the maximum float value (stored in the segment sizes for
                        # unsupported encodings) to the size. Hence, we replace the runtimes by max float whenever an
                        # encoding is actually not supported (prepare_vectors).
                        fastest_encoding_id = np.nanargmin(runtimes)
                    self.initial_configuration[i, j, k] = fastest_encoding_id
                    size += __segment_sizes_np[i, j, k, fastest_encoding_id]

                    data_type = __table_and_column_meta_data.loc[(i, k), 'DATA_TYPE']
                    for encoding_id in range(COUNT_CONF_OPTIONS):
                        if data_type not in encoding_selection_constants.SUPPORTED_DATA_TYPES[encoding_id]:
                            assert(np.isnan(sizes[encoding_id]))

                    weight_and_alternative = self.weight_function(sizes, runtimes, fastest_encoding_id, self.alpha)
                    if weight_and_alternative is not None:
                        heuristic_candidates.append([i, j, k, weight_and_alternative])


        self.initial_configuration_size = size

        # sort all alternative by decreasing weight
        self.heuristic_candidates_sorted = sorted(heuristic_candidates, key=lambda x: x[-1][0])


    def fit(self, budget):
        __segment_sizes_np = self.workload['segment_sizes_np']

        self.configuration = self.initial_configuration.copy()
        self.configuration_size = self.initial_configuration_size

        if PRINT_PLUGIN_INFORMATION:
            config_changes = list(static_config_changes)
            added_configurations = set() # we need to keep the order and only want the first config per column
            for config_change in self.heuristic_candidates_sorted:
                meta_line = self.workload['column_meta_data'].loc[config_change[0], config_change[2]]
                if f'{meta_line.TABLE_NAME},{meta_line.COLUMN_NAME}' not in added_configurations:
                    config_changes.append(('ACCESSED', meta_line.TABLE_NAME, meta_line.COLUMN_NAME,
                                           encoding_selection_constants.EncodingType(int(config_change[3][1])).name))
                    added_configurations.add(f'{meta_line.TABLE_NAME},{meta_line.COLUMN_NAME}')

        budget_reached = False

        for i, j, k, alternative in self.heuristic_candidates_sorted:
            if budget_reached:
                # once we are below the desired budget, stop.
                break

            alternative_encoding = alternative[1]
            current_size = __segment_sizes_np[i, j, k, int(self.configuration[i, j, k])]
            new_size = __segment_sizes_np[i, j, k, alternative_encoding]
            new_config_size = self.configuration_size + (new_size - current_size)

            if new_config_size <= budget:
                budget_reached = True

            self.configuration_size = new_config_size 
            self.configuration[i,j,k] = alternative_encoding


    def get_full_result(self):
        result_size = 0.0
        result_runtime = self.workload['all_dictionary_runtime']

        for i in range(self.workload['table_count']):
            for j in range(self.workload['table_dimensions'][i][0]):
                for k in range(self.workload['table_dimensions'][i][1]):
                    result_size += self.workload['segment_sizes_np'][i, j, k, int(self.configuration[i,j,k])]
                    runtime = self.workload['runtimes_numpy_matrix'][i, j, k, int(self.configuration[i,j,k])]
                    if not np.isnan(runtime):
                        result_runtime += runtime

        assert(result_size < self.configuration_size * 1.01 and result_size > self.configuration_size * 0.99)

        return ('Success', result_runtime, self.configuration_size, self.configuration, self.problem_time, self.solving_time)


    def get_final_configuration(self):
        return self.configuration


# We take various powers of the runtime (e.g., 0.5 and 2). For negative runtimes, we need to adapt negative
# runtimes in order to not yield nans (sqrt of neg value).
def adapt_negative_predictions(problematic_runtimes):
    min_pred = min(problematic_runtimes)
    if min_pred >= 0:
        return
    max_pred = max(problematic_runtimes)
    if max_pred < 0:
        # TODO: currently divide by 100, because we don't to have create use runtimes which have been negative before
        return (problematic_runtimes + (-1 * min_pred)) / 100

    # scale [min, max] to [0, max]
    return (problematic_runtimes + (-1 * min_pred)) * (max_pred / (max_pred - min_pred)) + 0.000001


def weight_like_ibm(sizes, runtimes, current_encoding_id, alpha):
    segment_not_accessed = np.isnan(np.sum(runtimes))
    ratios = []
    for runtime, size in zip(runtimes, sizes):
        # we first check for nan cases, which need to be handled with care as they might have different interpretations
        if np.isnan(size):
            # not supported data type
            ratios.append(-1)
            continue

        if np.isnan(runtime):
            # Two interpretations of a nan runtime: (i) never accessed (should be the case for all runtime values) or
            # (ii) encoding is not valid (e.g., the data type of the segment is not supported)
            if segment_not_accessed:
                # we shall yield values that will be picked up by the greedy heuristics because good compression for
                # never accessed segments is an easy win
                ratios.append(1 / size)
            else:
                ratios.append(-1)
            continue

        ratio = 1 / ((runtime**alpha) * size)
        ratios.append(ratio)

    max_ratio = max(ratios)
    max_ratio_encoding_id = ratios.index(max_ratio)

    return (max_ratio, max_ratio_encoding_id)


# Meant to be used with the default forward heuristic.
# Slower encodings (remember, we initially set the smallest encoding) yield negative weights.
def weight_by_runtime_change_to_size_ratio(sizes_, runtimes_, current_encoding_id, alpha):
    if np.isnan(runtimes_).all():
        return None

    if min(runtimes_) < 0.0:
        runtimes_ = adapt_negative_predictions(runtimes_)

    sizes = np.copy(sizes_)
    runtimes = np.power(runtimes_, alpha)

    current_runtime = runtimes[current_encoding_id]    

    if np.isnan(current_runtime):
        return None

    runtime_changes = runtimes - current_runtime
    ratios = runtime_changes / sizes

    return (np.nanmin(ratios), np.nanargmin(ratios))


def weight_by_size_to_runtime_change_ratio(sizes_, runtimes_, current_encoding_id, alpha):
    if np.isnan(runtimes_).all():
        return None

    if min(runtimes_) < 0.0:
        runtimes_ = adapt_negative_predictions(runtimes_)

    sizes = np.copy(sizes_)
    runtimes = np.power(runtimes_, alpha)

    current_size = sizes[current_encoding_id]

    size_changes = sizes - current_size

    ratios = size_changes / runtimes

    return (np.nanmin(ratios), np.nanargmin(ratios))


class LPCompressionSelection(BaseCompressionSelection):
    problem = None
    workload = None
    solving_time = 0.0
    problem_time = 0.0

    starting_configuration = None
    is_set_up = False

    lin_c_sel = None
    c = None
    d = None

    stored_base_constraints = None

    change_factors = None
    equal_slowdown = None
    equal_regression = None

    def __init__(self, workload_, starting_configuration = None, setup_at_init = True):
        self.workload = workload_
        self.starting_configuration = starting_configuration

        # Pickling the LP problem takes ~20s. Setting it up once is beneficial for sequential execution, but should be
        # avoided when processes are used.
        if setup_at_init:
            self.setup()
            self.is_set_up = True


    def setup(self):
        __table_count = self.workload['table_count']
        __table_dimensions = self.workload['table_dimensions']

        __runtimes_numpy = self.workload['runtimes_numpy']
        __base_runtime_dict_fwi = self.workload['all_dictionary_runtime']

        __unsupported_data_types_numpy = self.workload['unsupported_data_types_numpy']

        start_problem = datetime.datetime.now()
        self.lin_c_sel = pulp.LpProblem('Linear_Compression_Selection', pulp.LpMinimize)
        vars = [(i,j,k,l) for i in range(__table_count)
                for j in range(__table_dimensions[i][0])
                for k in range(__table_dimensions[i][1])
                for l in range(COUNT_CONF_OPTIONS)]
        self.c = pulp.LpVariable.dicts("c", vars, lowBound = 0, upBound = 1, cat = pulp.constants.LpBinary)
        self.d = pulp.LpVariable("d", cat = "Continuous")
        
        if self.starting_configuration is not None:
            for i in range(__table_count):
                for j in range(__table_dimensions[i][0]):
                    for k in range(__table_dimensions[i][1]):
                        for l in range(COUNT_CONF_OPTIONS):
                            if int(starting_configuration[i,j,k]) == l:
                                self.c[(i,j,k,l)].setInitialValue(1.0)
                            else:
                                self.c[(i,j,k,l)].setInitialValue(0.0)

        #########
        ######### objective: runtime
        #########

        summerands = [__base_runtime_dict_fwi]
        for row in __runtimes_numpy:
            summerands.append(self.c[(row[0], row[1], row[2], row[3])] * row[4])

        self.lin_c_sel += (pulp.lpSum(summerands))

        #########
        ######### variables
        #########

        ######### config consisting of values v with 0 <= v <= 1 (problem is integral)
        for i in range(__table_count):
            for j in range(__table_dimensions[i][0]):
                for k in range(__table_dimensions[i][1]):
                    self.lin_c_sel += pulp.lpSum(self.c[(i,j,k,l)] for l in range(COUNT_CONF_OPTIONS)) == 1

        ######### yield only encodings that support the column's data type
        for row in __unsupported_data_types_numpy:
            # row[0] is table id, row[1] is column_id
            for row_cluster in range(__table_dimensions[row[0]][0]):
                self.lin_c_sel += self.c[(row[0],row_cluster,row[1],row[2])] == 0


        # save constraints to be able to later re-add for another fit()
        self.stored_base_constraints = self.lin_c_sel.constraints.copy()


    def fit(self, budget, print_solving_time = False):
        if not self.is_set_up:
            self.setup()

        start_problem = datetime.datetime.now()

        __table_count = self.workload['table_count']
        __table_dimensions = self.workload['table_dimensions']

        __runtimes_numpy = self.workload['runtimes_numpy']
        __base_runtime_dict_fwi = self.workload['all_dictionary_runtime']

        __segment_sizes_np = self.workload['segment_sizes_np']
        __runtimes_numpy_query = self.workload['runtimes_numpy_query']
        __runtimes_dict_fwi_queries = self.workload['runtimes_dict_fwi_queries']

        __unsupported_data_types_numpy = self.workload['unsupported_data_types_numpy']

        # restore general base constraints
        self.lin_c_sel.constraints = self.stored_base_constraints.copy()

        ######### size being within the limit
        self.lin_c_sel += pulp.lpSum(__segment_sizes_np[i,j,k,l] * self.c[(i,j,k,l)] * (1 - int(np.any(np.all([i, k, l] == __unsupported_data_types_numpy, axis=1))))
                                for i in range(__table_count)
                                for j in range(__table_dimensions[i][0])
                                for k in range(__table_dimensions[i][1])
                                for l in range(COUNT_CONF_OPTIONS)) <= budget
        
        ######### queries that should not regress
        if self.change_factors is not None:
            assert __runtimes_numpy_query is not None
            for query_id, factor in self.change_factors.items():
                # We first obtain the estimated query runtime to get the actual difference to the desired query
                # runtime. Cannot do it directly as we are only having the changes per encoding.
                query_runtime = __runtimes_dict_fwi_queries[query_id]
                handle = factor * query_runtime - query_runtime

                constraint_summerands = []
                queries = __runtimes_numpy_query[query_id]
                for row in queries:
                    constraint_summerands.append(self.c[(row[0], row[1], row[2], row[3])] * row[4])

                self.lin_c_sel += pulp.lpSum(constraint_summerands) <= handle

        if self.equal_slowdown is not None:
            assert __runtimes_numpy_query is not None
            DICTIONARY_BASED_EQUAL_SLOWDOWN = True

            if DICTIONARY_BASED_EQUAL_SLOWDOWN:
                all_queries = set(__runtimes_dict_fwi_queries.keys())
                for query in all_queries:
                    q_runtime_dict = __runtimes_dict_fwi_queries[query]
                    q_changes = []
                    for row in  __runtimes_numpy_query[query]:
                        q_changes.append(self.c[(row[0], row[1], row[2], row[3])] * row[4])
             
                    self.lin_c_sel += (1 / self.equal_slowdown) * self.d * q_runtime_dict <= pulp.lpSum(q_changes) + q_runtime_dict
                    self.lin_c_sel += (1 * self.equal_slowdown) * self.d * q_runtime_dict >= pulp.lpSum(q_changes) + q_runtime_dict
            else:
                all_queries = set(__min_query_runtimes.keys())
                for query in all_queries:
                    min_q_runtime = __min_query_runtimes[query]
                    q_runtime_dict = __runtimes_dict_fwi_queries[query]
                    q_changes = []
                    for row in  __runtimes_numpy_query[query]:
                        q_changes.append(self.c[(row[0], row[1], row[2], row[3])] * row[4])

                    self.lin_c_sel += (1 / self.equal_slowdown) * self.d * min_q_runtime <= pulp.lpSum(q_changes) + q_runtime_dict
                    self.lin_c_sel += (1 * self.equal_slowdown) * self.d * min_q_runtime >= pulp.lpSum(q_changes) + q_runtime_dict
        ########### robustness end

        end_problem = datetime.datetime.now()

        if print_solving_time:
            print('Solving ... ', end='')
        start_solving = datetime.datetime.now()
        if self.solver == 'Gurobi':
            self.lin_c_sel.solve(pulp.GUROBI_CMD(options=[("Threads", self.thread_count), ("MIPGap", '1e-2'), ('TimeLimit', 600), ('OutputFlag', 0)]))
        elif self.solver == 'GLPK':
            self.lin_c_sel.solve(pulp.GLPK_CMD(msg=0, options=['--mipgap', '1e-2', '--tmlim', '600']))
        elif self.solver == 'Cbc':
            self.lin_c_sel.solve(pulp.PULP_CBC_CMD(timeLimit=600, msg=0, options=['ratioGAP', '1e-2', 'sec', '1800']))  # Changed gap for GitHub action
        elif self.solver == 'SCIP':
            self.lin_c_sel.solve(pulp.apis.scip_api.SCIP_CMD(timeLimit=600, msg=0, options=['-s', 'scip_settings.set']))
        elif self.solver == 'HiGHS':
            self.lin_c_sel.solve(pulp.HiGHS_CMD(timeLimit=600, msg=False))
        else:
            sys.exit(f'Solver {self.solver} not supported.')

        end_solving = datetime.datetime.now()

        self.problem_time = (end_problem - start_problem).total_seconds()
        self.solving_time = (end_solving - start_solving).total_seconds()
        if print_solving_time:
            print(f'done. \t Problem formulation: {self.problem_time*1000} ms. Solving: {self.solving_time*1000} ms.')

        self.problem = self.lin_c_sel


    def get_full_result(self):
        configuration = np.zeros((self.workload['table_count'],
                                  self.workload['max_row_clusters'],
                                  self.workload['max_column_count']))
        get_vars = lambda n: n[n.find('(')+1:n.find(')')].split(',_')

        size = 0
        for v in self.problem.variables():
            if not v.name.startswith('c'):
                # not part of the encoding configuration
                continue

            if v.varValue is None:
                # probably an infeasible problem
                continue

            table, row, column, encoding_id = map(int, get_vars(v.name))
            if v.varValue == 1.0:
                configuration[table, row, column] = encoding_id
                size += self.workload['segment_sizes_np'][table, row, column, encoding_id]

        status = pulp.constants.LpStatus[self.problem.status]
        if pulp.constants.LpStatus[self.problem.status] in ["Infeasible", "Not Solved"]:
            return (status, -1, -1, None, self.problem_time, self.solving_time)
        
        if self.problem.objective is None:
            print(f'Objective is empty. Status: {status}')
            return (status, -1, -1, None, self.problem_time, self.solving_time)

        runtime = pulp.value(self.problem.objective)
        return (status, runtime, size, configuration, self.problem_time, self.solving_time)


    def get_final_configuration(self):
        configuration = np.zeros((self.workload['table_count'],
                                  self.workload['max_row_clusters'],
                                  self.workload['max_column_count']))
        get_vars = lambda n: n[n.find('(')+1:n.find(')')].split(',_')

        print("Status:", pulp.constants.LpStatus[self.problem.status])
        if pulp.constants.LpStatus[self.problem.status] == "Infeasible":
            return None
        print("Runtime: ", pulp.value(self.problem.objective))
        size = 0
        for v in self.problem.variables():
            if not v.name.startswith('c'):
                # not part of the configuration
                continue

            if v.varValue is not None:
                table, row, column, encoding_id = map(int, get_vars(v.name))
                if not (0.0 <= v.varValue and v.varValue <= 1.0):
                    print('WARNING: value not in [0, 1], value is ', v.varValue)
                if v.varValue == 1.0:
                    configuration[table, row, column] = encoding_id
                    size += self.workload['segment_sizes_np'][table, row, column, encoding_id]

        print("Size: ", size)
        return configuration


    def get_query_runtimes(self):
        get_vars = lambda n: n[n.find('(')+1:n.find(')')].split(',_')

        __runtimes_numpy_query = self.workload['runtimes_numpy_query']
        assert __runtimes_numpy_query is not None

        # maps (table, row, col) > encoding_id
        final_configuration = np.zeros((self.workload['table_count'], self.workload['max_row_clusters'],
                                  self.workload['max_column_count'], COUNT_CONF_OPTIONS))

        for v in self.problem.variables():
            if not v.name.startswith('c'):
                # not part of the configuration
                continue

            table, row, column, encoding_id = map(int, get_vars(v.name))
            if not (0.0 <= v.varValue and v.varValue <= 1.0):
                print('WARNING: value not in [0, 1], value is ', v.varValue)
            final_configuration[table, row, column, encoding_id] = v.varValue

        dict_fwi_runtimes = self.workload['runtimes_dict_fwi_queries']
        query_runtime_changes = {}

        for query_id, queries in __runtimes_numpy_query.items():
            accumulated_change = 0.0

            for row in queries:
                accumulated_change += final_configuration[int(row[0]), int(row[1]), int(row[2]), int(row[3])] * row[4]

            query_runtime_changes[query_id] = accumulated_change + dict_fwi_runtimes[query_id]

        return query_runtime_changes


    def set_constraint(self, constraint_def: tuple):
        assert len(constraint_def) == 2
        constraint_title = constraint_def[0]
        constraint_value = constraint_def[1]

        if constraint_title == 'EqualRegression':
            self.equal_regression = constraint_value
        elif constraint_title == 'EqualSlowdown':
            self.equal_slowdown = constraint_value
        elif constraint_title == 'QuerySlowdown':
            self.change_factors = constraint_value
        else:
            sys.exit("Unexpected constraint type.")


    def print_information(self):
        df = pd.DataFrame()
        print("Status:", pulp.constants.LpStatus[self.problem.status])
        for v in self.problem.variables():
            if v.varValue is None:
                print(f'{v.name} is None.')
            else:
                print(f'{v.name} = {v.varValue}')
            df.append({'Name': v.name, 'Value': v.varValue}, ignore_index = True)
        print("Total Cost = ", pulp.value(self.problem.objective))

