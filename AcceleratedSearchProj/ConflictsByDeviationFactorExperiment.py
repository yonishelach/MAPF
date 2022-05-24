import csv
import math
import os
import random
import re

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes

from DatabaseInterface import greedily_generate_path_from_source_to_midpoint, \
    greedily_generate_path_from_midpoint_to_destination
from EnvironmentUtils import get_plan_conflicts


def get_midpoint_providing_shortest_path_smaller_than_arrival_time(warehouse, source_id, destination_id, arrival_time):
    warehouse_id = warehouse.warehouse_id
    file_path = f'./csv_files/warehouse_{warehouse_id}/midpoint_restricted_path_lengths/from_source_{source_id}' \
                f'_to_destination_{destination_id}.csv'
    midpoint_df = pd.read_csv(file_path)

    ideal_path_length = warehouse.sources[source_id].destination_distance[destination_id]
    deviation_factor = round(arrival_time / ideal_path_length, 2)
    if deviation_factor != round(deviation_factor):
        deviation_factor -= 0.01

    values_below_deviation_factor = midpoint_df.loc[midpoint_df.deviation_factor <= deviation_factor]

    sampled_midpoint = eval(values_below_deviation_factor.midpoint.sample().item())
    return sampled_midpoint


def generate_path_arriving_before_arrival_time(warehouse, source_id, destination_id, arrival_time):
    midpoint = get_midpoint_providing_shortest_path_smaller_than_arrival_time(warehouse, source_id, destination_id,
                                                                              arrival_time)

    path_to_midpoint = greedily_generate_path_from_source_to_midpoint(warehouse, source_id, midpoint)
    path_from_midpoint = greedily_generate_path_from_midpoint_to_destination(warehouse, destination_id, midpoint)
    path = path_to_midpoint + path_from_midpoint[1:]

    return path


def generate_path_for_given_arrival_time(warehouse, source_id, destination_id, arrival_time):
    path_without_waits = generate_path_arriving_before_arrival_time(warehouse, source_id, destination_id, arrival_time)
    path_length = len(path_without_waits)

    waits_at_source = [path_without_waits[0] for _ in range(arrival_time - path_length)]
    path_with_waits = waits_at_source + path_without_waits

    return path_with_waits


def generate_arrival_time(warehouse, source_id, destination_id, maximal_deviation_factor, deviation_factor):
    ideal_path_length = warehouse.sources[source_id].destination_distance[destination_id]
    if deviation_factor is None:
        deviation_factor = random.uniform(1, maximal_deviation_factor)

    deviation_factor = min(maximal_deviation_factor, deviation_factor)  # to prevent exceeding max deviation factor

    arrival_time = math.floor(ideal_path_length * deviation_factor)
    return arrival_time


def run_experiment(warehouse, number_of_agents, maximal_deviation_factor=1.1, deviation_factor=None):
    source_ids = [random.choice(range(warehouse.number_of_sources)) for _ in range(number_of_agents)]
    destination_ids = [random.choice(range(warehouse.number_of_destinations)) for _ in range(number_of_agents)]
    arrival_times = [generate_arrival_time(warehouse, source_ids[i], destination_ids[i], maximal_deviation_factor,
                                           deviation_factor)
                     for i in range(number_of_agents)]

    plan = [generate_path_for_given_arrival_time(warehouse, source_ids[i], destination_ids[i], arrival_times[i])
            for i in range(number_of_agents)]

    vertex_conflicts, swapping_conflicts = get_plan_conflicts(plan)

    routing_requests = [(source_ids[i], destination_ids[i]) for i in range(number_of_agents)]
    return routing_requests, plan, vertex_conflicts, swapping_conflicts


def create_experiment_directory_if_does_not_exist(warehouse_id):
    dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/data/"
    if not os.path.isdir(dir_path):
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)


def create_experiment_file_if_does_not_exist(warehouse_id, number_of_agents, deviation_factor):
    dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/data/"

    file_path = dir_path + f"{number_of_agents}-agents, deviation_factor-{deviation_factor}.csv"
    field_names = ['warehouse_id', 'routing_requests', 'plan', 'vertex_conflicts', 'swapping_conflicts']

    file_exists = os.path.isfile(file_path)
    if not file_exists:
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()


def create_experiment_file_hierarchy_if_does_not_exist(warehouse_id, number_of_agents, deviation_factor):
    create_experiment_directory_if_does_not_exist(warehouse_id)
    create_experiment_file_if_does_not_exist(warehouse_id, number_of_agents, deviation_factor)


def export_results_to_csv(warehouse_id, number_of_agents, deviation_factor, routing_requests, plan, vertex_conflicts,
                          swapping_conflicts):
    dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/data/"

    file_path = dir_path + f"{number_of_agents}-agents, deviation_factor-{deviation_factor}.csv"
    field_names = ['warehouse_id', 'routing_requests', 'plan', 'vertex_conflicts', 'swapping_conflicts']

    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)

        results_info = {'warehouse_id': warehouse_id, 'routing_requests': routing_requests, 'plan': plan,
                        'vertex_conflicts': vertex_conflicts, 'swapping_conflicts': swapping_conflicts}

        writer.writerow(results_info)


def run_experiments_to_generate_main_data_file_deviation_graph(warehouse, number_of_agents, number_of_samples, deviation_factor):
    warehouse_id = warehouse.warehouse_id
    create_experiment_file_hierarchy_if_does_not_exist(warehouse_id, number_of_agents, deviation_factor)

    for _ in range(number_of_samples):
        routing_requests, plan, vertex_conflicts, swapping_conflicts = run_experiment(warehouse, number_of_agents)
        export_results_to_csv(warehouse_id, number_of_agents, deviation_factor, routing_requests, plan,
                              vertex_conflicts,
                              swapping_conflicts)


def generate_number_of_conflicts_by_deviation_factor_data(warehouse_id, numbers_of_agents):
    data_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/data/"
    if not os.path.isdir(data_dir_path):
        print("No data available.")
        print(f"Please generate data via run_experiments_to_generate_main_data_file for warehouse with "
              f"warehouse_id={warehouse_id}")
        return

    for number_of_agents in numbers_of_agents:

        results = pd.DataFrame(columns=['deviation_factor', 'number_of_conflicts', 'number_of_agents'])

        for file_name in os.listdir(data_dir_path):
            file_path = data_dir_path + file_name

            raw_data = pd.read_csv(file_path)

            if int(file_name[:file_name.find('-')]) != number_of_agents:
                continue

            number_of_conflicts = np.mean([len(conflicts) for conflicts in raw_data.vertex_conflicts])
            deviation_factor = float(re.findall(r"-[0-9]\.[0-9]*\.csv", file_name)[0][1:-4])

            results = results.append({'deviation_factor': deviation_factor, 'number_of_conflicts': number_of_conflicts,
                                      'number_of_agents': number_of_agents}, ignore_index=True)

        results = results.sort_values(by='deviation_factor').set_index('deviation_factor')
        results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/results/"
        if not os.path.isdir(results_dir_path):
            os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

        results_file_path = results_dir_path + \
                            f'number_of_conflicts_by_deviation_factor_with_{number_of_agents}_agents.csv'
        results.to_csv(results_file_path)


def generate_number_of_conflicts_by_deviation_factor_visualization(warehouse_id, numbers_of_agents):
    for number_of_agents in numbers_of_agents:
        results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/results/"
        results_file_path = results_dir_path +\
                            f'number_of_conflicts_by_deviation_factor_with_{number_of_agents}_agents.csv'
        df = pd.read_csv(results_file_path)
        sns.lineplot(data=df, x='deviation_factor', y='number_of_conflicts', linewidth=3)

        x_max = plt.xlim()[1]
        plt.xlim(1, x_max)
        plt.suptitle(f'Number of conflicts by deviation factor - {number_of_agents} agents')
        plt.title(f'warehouse_id = {warehouse_id}')
        plt.savefig(results_dir_path + f'number_of_conflicts_by_deviation_factor_with_{number_of_agents}_agents.png')
        plt.show()


def generate_vertex_conflict_heatmap_data(warehouse):
    warehouse_id = warehouse.warehouse_id
    data_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/data/"
    if not os.path.isdir(data_dir_path):
        print("No data available.")
        print(f"Please generate data via run_experiments_to_generate_main_data_file for warehouse with "
              f"warehouse_id={warehouse_id}")
        return

    results = np.zeros(shape=(warehouse.width, warehouse.length))

    for file_name in os.listdir(data_dir_path):
        file_path = data_dir_path + file_name

        raw_data = pd.read_csv(file_path)
        for conflict_list in raw_data.vertex_conflicts:
            for conflict in eval(conflict_list):
                conflict_location = conflict[3]
                results[conflict_location[0], conflict_location[1]] += 1

    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/results/"
    if not os.path.isdir(results_dir_path):
        os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

    results_file_path = results_dir_path + 'vertex_conflict_heatmap_data.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file_path)


def generate_vertex_conflict_heatmap_visualization(warehouse_id):
    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/results/"
    results_file_path = results_dir_path + 'vertex_conflict_heatmap_data.csv'
    df = pd.read_csv(results_file_path, index_col='Unnamed: 0')
    sns.heatmap(data=df.loc[::-1])
    plt.suptitle('Vertex conflict, by location')
    plt.title(f'warehouse_id = {warehouse_id}', loc='left')

    plt.savefig(results_dir_path + 'vertex_conflict_heatmap_data.png')
    plt.show()


def generate_swapping_conflict_heatmap_data(warehouse):
    warehouse_id = warehouse.warehouse_id
    data_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/data/"
    if not os.path.isdir(data_dir_path):
        print("No data available.")
        print(f"Please generate data via run_experiments_to_generate_main_data_file for warehouse with "
              f"warehouse_id={warehouse_id}")
        return

    results = np.zeros(shape=(warehouse.width, warehouse.length))

    for file_name in os.listdir(data_dir_path):
        file_path = data_dir_path + file_name

        raw_data = pd.read_csv(file_path)
        for conflict_list in raw_data.swapping_conflicts:
            for conflict in eval(conflict_list):
                first_agent_conflict_location = conflict[3]
                second_agent_conflict_location = conflict[4]
                results[first_agent_conflict_location[0], first_agent_conflict_location[1]] += 1
                results[second_agent_conflict_location[0], second_agent_conflict_location[1]] += 1

    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/results/"
    if not os.path.isdir(results_dir_path):
        os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

    results_file_path = results_dir_path + 'swapping_conflict_heatmap_data.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file_path)


def generate_swapping_conflict_heatmap_visualization(warehouse_id):
    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/results/"
    results_file_path = results_dir_path + 'swapping_conflict_heatmap_data.csv'
    df = pd.read_csv(results_file_path, index_col='Unnamed: 0')
    sns.heatmap(data=df.loc[::-1])
    plt.suptitle('Swapping conflict, by location')
    plt.title(f'warehouse_id = {warehouse_id}', loc='left')

    plt.savefig(results_dir_path + 'swapping_conflict_heatmap_data.png')
    plt.show()


def generate_plan_heatmap_data(warehouse):
    warehouse_id = warehouse.warehouse_id
    data_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/data/"
    if not os.path.isdir(data_dir_path):
        print("No data available.")
        print(f"Please generate data via run_experiments_to_generate_main_data_file for warehouse with "
              f"warehouse_id={warehouse_id}")
        return

    results = np.zeros(shape=(warehouse.width, warehouse.length))

    for file_name in os.listdir(data_dir_path):
        file_path = data_dir_path + file_name

        raw_data = pd.read_csv(file_path)
        for plan in raw_data.plan:
            for path in eval(plan):
                for vertex in path:
                    results[vertex] += 1

    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/results/"
    if not os.path.isdir(results_dir_path):
        os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

    results_file_path = results_dir_path + 'plan_heatmap_data.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file_path)


def generate_plan_heatmap_visualization(warehouse_id):
    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments/conflicts_by_deviation_factor/results/"
    results_file_path = results_dir_path + 'plan_heatmap_data.csv'
    df = pd.read_csv(results_file_path, index_col='Unnamed: 0')
    sns.heatmap(data=df.loc[::-1])
    plt.suptitle('Plan heatmap by location')
    plt.title(f'warehouse_id = {warehouse_id}', loc='left')

    plt.savefig(results_dir_path + 'plan_heatmap_data.png')
    plt.show()
