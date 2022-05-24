import csv
import math
import os
import random

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes

from DatabaseInterface import greedily_generate_path_from_source_to_midpoint, \
    greedily_generate_path_from_midpoint_to_destination
from EnvironmentUtils import get_plan_conflicts
from matplotlib.colors import LogNorm


MAX_DEVIATION_FACTOR = 1.7


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


def generate_arrival_time(warehouse, source_id, destination_id, maximal_deviation_factor):
    ideal_path_length = warehouse.sources[source_id].destination_distance[destination_id]
    deviation_factor = random.uniform(1, maximal_deviation_factor)

    arrival_time = math.floor(ideal_path_length * deviation_factor)
    return arrival_time


def run_experiment(warehouse, number_of_agents, maximal_deviation_factor=MAX_DEVIATION_FACTOR):
    # source_id = int(warehouse.number_of_sources / 2)
    # destination_id = int(warehouse.number_of_sources / 2)

    source_ids = [np.random.randint(warehouse.number_of_sources) for _ in range(number_of_agents)]
    destination_ids = [np.random.randint(warehouse.number_of_destinations) for _ in range(number_of_agents)]
    arrival_times = [generate_arrival_time(warehouse, source_ids[i], destination_ids[i], maximal_deviation_factor)
                     for i in range(number_of_agents)]

    plan = [generate_path_for_given_arrival_time(warehouse, source_ids[i], destination_ids[i], arrival_times[i])
            for i in range(number_of_agents)]

    vertex_conflicts, swapping_conflicts = get_plan_conflicts(plan)

    routing_requests = [(source_ids[i], destination_ids[i]) for i in range(number_of_agents)]
    return routing_requests, plan, vertex_conflicts, swapping_conflicts


def create_experiment_directory_if_does_not_exist(warehouse_id):
    dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
               f"/single_source_destination_conflicts_by_number_of_agents/data/"
    if not os.path.isdir(dir_path):
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)


def create_experiment_file_if_does_not_exist(warehouse_id, number_of_agents):
    dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
               f"/single_source_destination_conflicts_by_number_of_agents/data/"

    file_path = dir_path + f"{number_of_agents}_agents.csv"
    field_names = ['warehouse_id', 'routing_requests', 'plan', 'vertex_conflicts', 'swapping_conflicts']

    file_exists = os.path.isfile(file_path)
    if not file_exists:
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()


def create_experiment_file_hierarchy_if_does_not_exist(warehouse_id, number_of_agents):
    create_experiment_directory_if_does_not_exist(warehouse_id)
    create_experiment_file_if_does_not_exist(warehouse_id, number_of_agents)


def export_results_to_csv(warehouse_id, number_of_agents, routing_requests, plan, vertex_conflicts, swapping_conflicts):
    dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
               f"/single_source_destination_conflicts_by_number_of_agents/data/"

    file_path = dir_path + f"{number_of_agents}_agents.csv"
    field_names = ['warehouse_id', 'routing_requests', 'plan', 'vertex_conflicts', 'swapping_conflicts']

    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)

        results_info = {'warehouse_id': warehouse_id, 'routing_requests': routing_requests, 'plan': plan,
                        'vertex_conflicts': vertex_conflicts, 'swapping_conflicts': swapping_conflicts}

        writer.writerow(results_info)


def run_experiments_to_generate_main_data_file(warehouse, number_of_agents, number_of_samples):
    warehouse_id = warehouse.warehouse_id
    create_experiment_file_hierarchy_if_does_not_exist(warehouse_id, number_of_agents)

    for _ in range(number_of_samples):
        routing_requests, plan, vertex_conflicts, swapping_conflicts = run_experiment(warehouse, number_of_agents)
        export_results_to_csv(warehouse_id, number_of_agents, routing_requests, plan, vertex_conflicts,
                              swapping_conflicts)


def generate_conflict_probability_by_number_of_agents_data(warehouse_id):
    data_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                    f"/single_source_destination_conflicts_by_number_of_agents/data/"
    if not os.path.isdir(data_dir_path):
        print("No data available at", data_dir_path)
        print(f"Please generate data via run_experiments_to_generate_main_data_file for warehouse with "
              f"warehouse_id={warehouse_id}")
        return

    results = pd.DataFrame(columns=['number_of_agents', 'conflict_probability'])

    for file_name in os.listdir(data_dir_path):
        file_path = data_dir_path + file_name

        raw_data = pd.read_csv(file_path)
        vertex_conflicts = [len(eval(conflict)) > 0 for conflict in raw_data.vertex_conflicts]
        swapping_conflicts = [len(eval(conflict)) > 0 for conflict in raw_data.swapping_conflicts]
        is_conflict_in_sample = [vertex_conflicts[i] or swapping_conflicts[i] for i in range(len(vertex_conflicts))]

        number_of_samples = raw_data.shape[0]
        number_of_samples_with_conflicts = sum(is_conflict_in_sample)
        conflict_probability = round(number_of_samples_with_conflicts / number_of_samples, 2)
        number_of_agents = int(file_name[:file_name.find('_')])

        results = results.append({'number_of_agents': number_of_agents, 'conflict_probability': conflict_probability},
                                 ignore_index=True)

    results = results.sort_values(by='number_of_agents').set_index('number_of_agents')
    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                       f"/single_source_destination_conflicts_by_number_of_agents/results/"
    if not os.path.isdir(results_dir_path):
        os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

    results_file_path = results_dir_path + 'conflict_probability_by_number_of_agents.csv'
    results.to_csv(results_file_path)
    print("Conflict probability by number of agents data saved to:", results_file_path)


def generate_conflict_probability_by_number_of_agents_visualization(warehouse_id):
    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                       f"/single_source_destination_conflicts_by_number_of_agents/results/"
    if not os.path.isdir(results_dir_path):
        os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

    results_file_path = results_dir_path + 'conflict_probability_by_number_of_agents.csv'
    df = pd.read_csv(results_file_path)
    sns.lineplot(data=df, x='number_of_agents', y='conflict_probability', linewidth=3)

    x_max = plt.xlim()[1]
    plt.plot([0, x_max], [0.1, 0.1], linestyle='dashed', label='Low probability')
    plt.xlim(0, x_max)
    plt.ylim(0, 1)
    plt.suptitle('Conflict probability, by number of agents')
    plt.title(f'warehouse_id = {warehouse_id}', loc='left')
    plt.legend()

    image_file_path = results_dir_path + 'conflict_probability_by_number_of_agents.png'
    plt.savefig(image_file_path)
    print("Conflict probability by number of agents image saved to:", image_file_path)
    plt.show()


def generate_vertex_conflict_heatmap_data(warehouse):
    warehouse_id = warehouse.warehouse_id
    data_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                    f"/single_source_destination_conflicts_by_number_of_agents/data/"
    if not os.path.isdir(data_dir_path):
        print("No data available at", data_dir_path)
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

    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                       f"/single_source_destination_conflicts_by_number_of_agents/results/"
    if not os.path.isdir(results_dir_path):
        os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

    results_file_path = results_dir_path + 'vertex_conflict_heatmap_data.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file_path)
    print("Vertex conflict heatmap data saved to:", results_file_path)


def generate_vertex_conflict_heatmap_visualization(warehouse_id, log_scale=False):
    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                       f"/single_source_destination_conflicts_by_number_of_agents/results/"
    if not os.path.isdir(results_dir_path):
        os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

    results_file_path = results_dir_path + 'vertex_conflict_heatmap_data.csv'
    kwargs = dict()
    if log_scale:
        kwargs["norm"] = LogNorm()
    df = pd.read_csv(results_file_path, index_col='Unnamed: 0')
    sns.heatmap(data=df.loc[::-1], **kwargs)
    plt.suptitle('Vertex conflict, by location' + (' (logscale)' if log_scale else ''))
    plt.title(f'warehouse_id = {warehouse_id}', loc='left')

    image_file_path = results_dir_path + 'vertex_conflict_heatmap_data' + ('_log' if log_scale else '') + '.png'
    plt.savefig(image_file_path)
    print("Vertex conflict heatmap image saved to:", image_file_path)
    plt.show()


def generate_swapping_conflict_heatmap_data(warehouse):
    warehouse_id = warehouse.warehouse_id
    data_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                    f"/single_source_destination_conflicts_by_number_of_agents/data/"
    if not os.path.isdir(data_dir_path):
        print("No data available at", data_dir_path)
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

    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                       f"/single_source_destination_conflicts_by_number_of_agents/results/"
    if not os.path.isdir(results_dir_path):
        os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

    results_file_path = results_dir_path + 'swapping_conflict_heatmap_data.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file_path)
    print("Swapping conflict heatmap data saved to:", results_file_path)


def generate_swapping_conflict_heatmap_visualization(warehouse_id, log_scale=False):
    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                       f"/single_source_destination_conflicts_by_number_of_agents/results/"
    if not os.path.isdir(results_dir_path):
        os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

    results_file_path = results_dir_path + 'swapping_conflict_heatmap_data.csv'
    df = pd.read_csv(results_file_path, index_col='Unnamed: 0')
    kwargs = dict()
    if log_scale:
        kwargs["norm"] = LogNorm()
    sns.heatmap(data=df.loc[::-1], **kwargs)
    plt.suptitle('Swapping conflict, by location' + (' (logscale)' if log_scale else ''))
    plt.title(f'warehouse_id = {warehouse_id}', loc='left')

    image_file_path = results_dir_path + 'swapping_conflict_heatmap_data' + ('_log' if log_scale else '') + '.png'
    plt.savefig(image_file_path)
    print("Swapping conflict heatmap image saved to:", image_file_path)
    plt.show()


def generate_plan_heatmap_data(warehouse):
    warehouse_id = warehouse.warehouse_id
    data_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                    f"/single_source_destination_conflicts_by_number_of_agents/data/"
    if not os.path.isdir(data_dir_path):
        print("No data available at", data_dir_path)
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

    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                       f"/single_source_destination_conflicts_by_number_of_agents/results/"
    if not os.path.isdir(results_dir_path):
        os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

    results_file_path = results_dir_path + 'plan_heatmap_data.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file_path)
    print("Plan heatmap data saved to:", results_file_path)


def generate_plan_heatmap_visualization(warehouse_id, log_scale=False):
    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                       f"/single_source_destination_conflicts_by_number_of_agents/results/"
    if not os.path.isdir(results_dir_path):
        os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

    results_file_path = results_dir_path + 'plan_heatmap_data.csv'
    df = pd.read_csv(results_file_path, index_col='Unnamed: 0')
    kwargs = dict()
    kwargs["cmap"] = "Blues"
    if log_scale:
        kwargs["norm"] = LogNorm()
    sns.heatmap(data=df.loc[::-1], **kwargs)
    plt.suptitle('Plan heatmap by location' + (' (logscale)' if log_scale else ''))
    plt.title(f'warehouse_id = {warehouse_id}', loc='left')

    image_file_path = results_dir_path + 'plan_heatmap_data' + ('_log' if log_scale else '') + '.png'
    plt.savefig(image_file_path)
    print("Plan heatmap image saved to:", image_file_path)
    plt.show()


def sample_path_database(warehouse_id, number_of_paths=10):
    data_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                    f"/single_source_destination_conflicts_by_number_of_agents/data/"
    if not os.path.isdir(data_dir_path):
        print("No data available at", data_dir_path)
        print(f"Please generate data via run_experiments_to_generate_main_data_file for warehouse with "
              f"warehouse_id={warehouse_id}")
        return

    file_name = "1_agents.csv"
    file_path = data_dir_path + file_name

    file_exists = os.path.isfile(file_path)
    if not file_exists:
        print("File does not exist:" + file_path)
        return []

    output = []
    raw_data = pd.read_csv(file_path)
    for plan in raw_data.plan:
        output.append(eval(plan)[0])

    return output


def visualize_plan_metro_map(warehouse, plan):
    warehouse_id = warehouse.warehouse_id
    warehouse.plot_layout()
    plt.suptitle('Plan as metro map')
    plt.title(f'warehouse_id = {warehouse_id}', loc='left')

    random.shuffle(plan)
    for i, route in enumerate(plan):
        if i > 10:
            continue

        x_coordinates = [coordinate[0] for coordinate in route]
        y_coordinates = [coordinate[1] for coordinate in route]

        plt.plot(y_coordinates, x_coordinates, linewidth=(7 - (i / 2)))

    results_dir_path = f"./csv_files/warehouse_{warehouse_id}/experiments" \
                       f"/single_source_destination_conflicts_by_number_of_agents/results/"
    if not os.path.isdir(results_dir_path):
        os.makedirs(os.path.dirname(results_dir_path), exist_ok=True)

    results_file_path = results_dir_path + 'plan_metro_map.png'

    plt.savefig(results_file_path)
    plt.show()


def generate_metro_map_visualization(warehouse):
    warehouse_id = warehouse.warehouse_id
    plan = sample_path_database(warehouse_id)
    visualize_plan_metro_map(warehouse, plan)
