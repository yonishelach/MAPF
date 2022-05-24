import os
from typing import List, Tuple, Dict
import csv

from matplotlib import pyplot as plt

import Warehouse
import random
import pandas as pd
from EnvironmentUtils import get_source_id_from_route, get_destination_id_from_route, \
    get_all_source_and_destination_combinations, count_plan_conflicts, is_agent_plan_conflicting_with_plan
import operator as op
from functools import reduce
from math import sqrt

import ExampleGeneration

# from ExampleGeneration import generate_example

SORT_BY = 'deviation_factor'
ROUTE_SAMPLING_TRIES = 3

DATABASE_GENERATION_ALGORITHMS = ["sample_database", "sample_database_avoiding_conflicts"]


def sort_rows_and_remove_duplicates_in_csv(file_name: str) -> None:
    """ saving the file_name with updated csv - sorted by 'SORT_BY' value, and without duplicates

    Args:
        file_name (str): a name of a file which exists
    """
    data = pd.read_csv(file_name, index_col=False, na_values='')
    sorted_csv = data.sort_values(by=[SORT_BY], ascending=True)
    sorted_csv = sorted_csv.drop_duplicates()
    x = len(sorted_csv.index)
    y = len(data.index)
    sorted_csv.to_csv(file_name, index=False)


# def calculate_deviation_cost(route: List, source_id: int, destination_id: int, warehouse: Warehouse) -> int:
#     """ calculates the cost of deviation
#     """
#     if len(route) == 0:
#         return len(route)
#     mid_point_of_shortest_route_coordinates = warehouse.sources_to_destinations_mid_point[source_id][destination_id]
#     sum_of_euclidian_distance_to_mid_point = 0.0
#     for node_coordinates in route:
#         sum_of_euclidian_distance_to_mid_point += sqrt(
#             pow(node_coordinates[0] - mid_point_of_shortest_route_coordinates[0], 2) +
#             pow(node_coordinates[1] - mid_point_of_shortest_route_coordinates[1], 2))
#     average_euclidian_distance_to_mid_point = sum_of_euclidian_distance_to_mid_point / len(route)
#     deviation_cost = average_euclidian_distance_to_mid_point - \
#                      warehouse.sources_to_destinations_average_euclidian_distance_to_mid_point[source_id][
#                          destination_id]
#     print("deviation_cost", deviation_cost)
#     return deviation_cost


# def nCk(n, r):
#     """ claculatimg combinatorics: n chose r
#     """
#     if n == 0:
#         return 0
#
#     r = min(r, n - r)
#     numer = reduce(op.mul, range(n, n - r, -1), 1)
#     denom = reduce(op.mul, range(1, r + 1), 1)
#     return numer // denom


# def create_number_of_collision(rows: List, field_names: List, column_length: int) -> int:
#     """ calculates the total number of collision
#
#     Args:
#         rows (List): List of all the routes
#         column_length (int): max length of columns
#     Returns:
#         int: number of collision  (1: switching places; 2: standing in same place in same time)
#     """
#     # column length = n; number of rows = m
#
#     same_place_count = 0
#     same_place_count_time_i = 0
#     # checking collisions inside columns - O(m^2 * n)
#     for time_i in range(column_length):
#         same_place_count_time_i = 0
#         for row in range(len(rows)):
#             if rows[row][field_names[time_i]] == '' or time_i in [0, 1, 2]:
#                 continue
#             row_below = row + 1
#             while row_below < len(rows):
#                 if rows[row_below][field_names[time_i]] == '':
#                     row_below = row_below + 1
#                     continue
#                 if rows[row][field_names[time_i]] == rows[row_below][field_names[time_i]]:
#                     same_place_count_time_i = same_place_count_time_i + 1
#                 row_below = row_below + 1
#
#         same_place_count = same_place_count + nCk(same_place_count_time_i, 2)  # n chose k
#
#     # checking collisions with changing places - O(n^2 * m)
#     changing_places_count = 0
#     for row in range(len(rows) - 1):
#         for time_i in range(column_length - 1):
#             if rows[row][field_names[time_i]] == '' or \
#                     rows[row + 1][field_names[time_i]] == '' or time_i in [0, 1, 2]:
#                 continue
#             row_below = row + 1
#             while row_below < len(rows):
#                 if rows[row_below][field_names[time_i]] == '' or \
#                         rows[row_below][field_names[time_i + 1]] == '':
#                     row_below = row_below + 1
#                     continue
#                 if rows[row][field_names[time_i]] == rows[row_below][field_names[time_i + 1]] and \
#                         rows[row][field_names[time_i + 1]] == rows[row_below][field_names[time_i]]:
#                     changing_places_count = changing_places_count + 1
#                 row_below = row_below + 1
#
#     return same_place_count + changing_places_count


# def create_length(rows: List) -> int:
#     """
#     Args:
#         rows (List): list of dictyonarys - each row is a dictionary of source to target route
#
#     Returns:
#         int: sum of total length of the routes
#     """
#     length = 0
#     for row in rows:
#         row['Algorithm Name'] = ''
#         row['Source Id'] = ''
#         row['Destination Id'] = ''
#         row = dict((k, v) for k, v in row.items() if v != '')
#         length = length + len(row)
#     return length


# def create_row_from_tupple(source: int, target: int, field_names: List, warehouse: Warehouse) -> Dict:
#     """ Generate a path from source to target from existing csv file
#
#     Args:
#         source (int): Source Id
#         target (int): Target Id
#         field_names (List) : a list of the headers
#         warehouse (Warehouse)
#     Returns:
#         Dict: Returns random route from source to target
#     """
#
#     file_name = './csv_files/warehouse_{}/paths/paths_from_{}_to_{}.csv'.format(warehouse.warehouse_id, source,
#                                                                                   target)
#     file_exists = os.path.isfile(file_name)
#     if not file_exists:
#         return None
#     data = import_routes_from_csv(file_name, field_names)
#     random_route_number = random.randint(0, (len(data) - 1))
#     return data[random_route_number]


# def export_all_routes_to_csv(warehouse: Warehouse, source_dest_list: List) -> None:
#     """ Creates a .csv file of routes for all the souce,dest tupples that were given
#
#     Args:
#         source_dest_list (List): List of tupples of wanted (source, dest) routes
#     """
#
#     field_names = create_header_routes_csv(warehouse)
#     file_name = './csv_files/warehouse_{}/all_chosen_routes.csv'.format(warehouse.warehouse_id)
#     file_exists = os.path.isfile(file_name)
#     os.makedirs(os.path.dirname(file_name), exist_ok=True)
#     with open(file_name, 'w', newline='') as f:
#         writer = csv.writer(f)
#         rows = []
#         for tupple in source_dest_list:
#             row = create_row_from_tupple(tupple[0], tupple[1], field_names, warehouse)
#             rows.append(row)
#         writer = csv.DictWriter(f, fieldnames=field_names)
#         writer.writerows(rows)
#         total_length = create_length(rows)
#         columns_length = warehouse.length + warehouse.width
#         number_of_collisions = create_number_of_collision(rows, field_names, columns_length)
#         writer = csv.writer(f)
#         total_length_str = 'Total Lenth:' + str(total_length)
#         number_of_collisions_Str = 'Total collisions:' + str(number_of_collisions)
#         calc_values = [total_length_str, number_of_collisions_Str]
#         writer.writerow(calc_values)
#
#         sort_rows_and_remove_duplicates_in_csv(file_name)


def import_routes_from_csv(csv_file: str, field_names: List = None) -> List:
    """ 

    Args:
        csv_file (str): The name of the csv file

    Returns:
        [List]: A list of all the routes from the csv_file
    """
    # check first if file exists
    file_exists = os.path.isfile(csv_file)
    if not file_exists:
        return []
    with open(csv_file, newline='') as f:
        if field_names is None:
            reader = csv.reader(f)
            file_content = list(reader)
            file_content_without_header = file_content[1:]
            routes_in_string_format = [row[5:] for row in file_content_without_header if row]
            routes = [[eval(tupple) for tupple in row] for row in routes_in_string_format if tupple]
        else:
            with open(csv_file, newline='') as f:
                reader = csv.DictReader(f)
                routes = []
                for row in reader:
                    dict_row = {}
                    for field_name in field_names:
                        dict_row[field_name] = row[field_name]
                    routes.append(dict_row)
        return routes


def import_routing_request_routes_from_database(warehouse, routing_request):
    warehouse_id = warehouse.warehouse_id
    source_id, destination_id = routing_request[0][0], routing_request[0][1]
    file_path = './csv_files/warehouse_{}/paths/paths_from_{}_to_{}.csv'.format(warehouse_id, source_id,
                                                                                destination_id)

    return import_routes_from_csv(file_path)


def sample_routing_request_route_from_database(warehouse, routing_request):
    warehouse_id = warehouse.warehouse_id
    source_id, destination_id = routing_request[0][0], routing_request[0][1]
    file_path = './csv_files/warehouse_{}/paths/paths_from_{}_to_{}.csv'.format(warehouse_id, source_id,
                                                                                destination_id)

    file_exists = os.path.isfile(file_path)
    if not file_exists:
        return []

    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        file_content = list(reader)
        file_content_without_header = file_content[1:]
        sampled_row_in_string_format = random.choice(file_content_without_header)
        sampled_route_in_string_format = sampled_row_in_string_format[5:]
        route = [eval(tupple) for tupple in sampled_route_in_string_format if tupple]

        return route


def create_header_routes_csv(warehouse: Warehouse, route=None) -> List:
    columns_length = warehouse.length + warehouse.width if not route else len(route)
    field_names = ['Warehouse Id', 'Source Id', 'Destination Id', SORT_BY]
    for i in range(columns_length):
        field_name = 'Time = {}'.format(i)
        field_names.append(field_name)
    return field_names


def calculate_deviation_factor(route, source_id, destination_id, warehouse):
    ideal_path_length = warehouse.sources[source_id].destination_distance[destination_id]
    input_path_length = len(route)

    return round(input_path_length / ideal_path_length, 2)


def create_route_information_dictionary(warehouse: Warehouse, algorithm_name: str, source_id: int, destination_id: int,
                                        field_names: List,
                                        route: List) -> Dict:
    """

    Args:
        algorithm_name (str): Name of the algorithm  
        source_id ([int]): Id of the vertex source
        destination_id ([int]): Id of the vertex destination
        field_names (List): List of strings - header of the row in a table
        route (List): route of touples from source to destination

    Returns:
        Dict: [Table row of {header_type_1:value_1, ... header_type_n:value_n}]
    """

    deviation_factor = calculate_deviation_factor(route, source_id, destination_id, warehouse)
    row = {'Warehouse Id': warehouse.warehouse_id, 'Source Id': source_id, 'Destination Id': destination_id,
           SORT_BY: deviation_factor}
    write_from_column = len(row)
    for i in range(len(route)):
        row[field_names[i + write_from_column]] = route[i]

    return row


def create_header_warehouse_csv(warehouse: Warehouse) -> List:
    """

    Args:
        warehouse (Warehouse):

    Returns:
        List: Headers of the table
    """
    field_names = ['Warehouse Id', 'Width', 'Length', 'Number Of Sources', 'Number Of Destinations',
                   'Static Obstacle Width', 'Static Obstacle Length']
    for i in range(len(warehouse.sources)):
        field_name = 'Source {}'.format(i + 1)
        field_names.append(field_name)
    for i in range(len(warehouse.destinations)):
        field_name = 'Destination {}'.format(i + 1)
        field_names.append(field_name)
    for i in range(len(warehouse.static_obstacle_layout)):
        field_name = 'Obstacle {} bot left corner'.format(i + 1)
        field_names.append(field_name)

    return field_names


def create_row_warehouse_csv(warehouse: Warehouse) -> Dict:
    """

    Args:
        warehouse (Warehouse):

    Returns:
        Dict: A row for the warehouse csv with the values for the table
    """
    row = {}
    for i in range(len(warehouse.sources)):
        header = 'Source {}'.format(i + 1)
        row[header] = warehouse.sources[i].coordinates
    for i in range(len(warehouse.destinations)):
        header = 'Destination {}'.format(i + 1)
        row[header] = warehouse.destinations[i].coordinates
    for i in range(len(warehouse.static_obstacle_layout)):
        header = 'Obstacle {} bot left corner'.format(i + 1)
        row[header] = warehouse.static_obstacle_layout[i]
    row['Warehouse Id'] = warehouse.warehouse_id
    row['Width'] = warehouse.width
    row['Length'] = warehouse.length
    row['Number Of Sources'] = warehouse.number_of_sources
    row['Number Of Destinations'] = warehouse.number_of_destinations
    row['Static Obstacle Width'] = warehouse.static_obstacle_width
    row['Static Obstacle Length'] = warehouse.static_obstacle_length
    return row


def write_route_information_to_file(file_name, warehouse, algorithm_name, source_id, destination_id, route):
    with open(file_name, 'a', newline='') as f:
        field_names = create_header_routes_csv(warehouse, route)
        writer = csv.DictWriter(f, fieldnames=field_names)

        row = create_route_information_dictionary(warehouse, algorithm_name, source_id, destination_id, field_names,
                                                  route)
        writer.writerow(row)


def create_routes_file_if_does_not_exist(file_name, warehouse):
    file_exists = os.path.isfile(file_name)
    if not file_exists:
        with open(file_name, 'w', newline='') as f:
            field_names = create_header_routes_csv(warehouse)
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()


def export_warehouse_information_to_csv(warehouse: Warehouse):
    """ Generates a .csv file using the above input

    Args:
        warehouse (Warehouse)
    """
    file_name = './csv_files/warehouse_{}/warehouse_{}_layout.csv'.format(warehouse.warehouse_id,
                                                                          warehouse.warehouse_id)
    file_exists = os.path.isfile(file_name)

    with open(file_name, 'w', newline='') as f:
        field_names = create_header_warehouse_csv(warehouse)
        writer = csv.DictWriter(f, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()

        row = create_row_warehouse_csv(warehouse)
        writer.writerow(row)


def export_warehouse_layout_image(warehouse):
    plot_grid = True
    warehouse.plot_layout(plot_grid)
    warehouse_id = warehouse.warehouse_id
    plt.savefig(f'./csv_files/warehouse_{warehouse_id}/warehouse_{warehouse_id}_layout.png')


def create_warehouse_dir_if_does_not_exist(warehouse):
    target_dir = "./csv_files/warehouse_{}/".format(warehouse.warehouse_id)

    if not os.path.isdir(target_dir):
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        export_warehouse_information_to_csv(warehouse)
        export_warehouse_layout_image(warehouse)


def create_warehouse_dir(warehouse):
    target_dir = "./csv_files/warehouse_{}/".format(warehouse.warehouse_id)

    if not os.path.isdir(target_dir):
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    export_warehouse_information_to_csv(warehouse)
    export_warehouse_layout_image(warehouse)


def export_plan_to_csv(algorithm_name, plan, warehouse):
    warehouse_id = warehouse.warehouse_id

    target_dir = f'./csv_files/warehouse_{warehouse_id}/paths/'
    if not os.path.isdir(target_dir):
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    for route in plan:
        source_id = get_source_id_from_route(warehouse, route)
        destination_id = get_destination_id_from_route(warehouse, route)
        file_name = './csv_files/warehouse_{}/paths/paths_from_{}_to_{}.csv'.format(warehouse_id, source_id,
                                                                                    destination_id)
        create_routes_file_if_does_not_exist(file_name, warehouse)
        write_route_information_to_file(file_name, warehouse, algorithm_name, source_id, destination_id, route)

    sort_rows_and_remove_duplicates_in_csv(file_name)


def build_routes_for_database(warehouse, algorithm_name="MPR_WS"):
    all_source_and_destination_combinations = get_all_source_and_destination_combinations(warehouse)

    for combination in all_source_and_destination_combinations:
        plan, running_time, routing_requests = ExampleGeneration.generate_example(warehouse, algorithm_name,
                                                                                  [combination])
        export_plan_to_csv(algorithm_name, plan, warehouse)


def sample_routing_request_plan_from_database(warehouse, routing_requests):
    plan = []
    agent_scheduling_by_source = [set() for _ in range(warehouse.number_of_sources)]

    for request in routing_requests:
        agent_plan = sample_routing_request_route_from_database(warehouse, [request])
        time = 0
        agent_source = request[0]
        agent_source_coordinates = warehouse.sources[agent_source].coordinates
        scheduled = False
        while not scheduled:
            agent_source_busy = time in agent_scheduling_by_source[agent_source]
            if not agent_source_busy:
                random_wait_at_source = random.choice([True, False])

                if not random_wait_at_source:
                    agent_scheduling_by_source[agent_source].add(time)
                    scheduled = True

            else:
                agent_plan.insert(0, agent_source_coordinates)
                time += 1
        plan.append(agent_plan)

    return plan


def create_header_for_tagged_examples_csv(warehouse: Warehouse, route=None) -> List:
    columns_length = warehouse.length + warehouse.width if not route else len(route)
    field_names = ['Source Id', 'Destination Id']

    for i in range(columns_length):
        field_name = 'Time = {}'.format(i)
        field_names.append(field_name)
    return field_names


def create_tagged_examples_file_if_does_not_exist(warehouse, file_name):
    file_exists = os.path.isfile(file_name)
    if not file_exists:
        with open(file_name, 'w', newline='') as f:
            field_names = create_header_for_tagged_examples_csv(warehouse)
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()


def build_tagged_examples_for_database(warehouse, num_of_waves=1, iterations=1):
    warehouse_id = warehouse.warehouse_id
    # dir_name = generate_random_name(num_of_waves)
    file_name = './csv_files/warehouse_{}/tagged_examples.csv'.format(warehouse_id)

    create_tagged_examples_file_if_does_not_exist(warehouse, file_name)

    for _ in range(iterations):
        plan = ExampleGeneration.generate_example(warehouse, "sample_database")[0]
        vertex_conflicts, swapping_conflicts = count_plan_conflicts(plan)


def sample_routing_database_avoiding_conflicts(warehouse, routing_requests):
    plan = []
    agent_scheduling_by_source = [set() for _ in range(warehouse.number_of_sources)]

    for request in routing_requests:
        time = 0
        agent_source = request[0]
        agent_source_coordinates = warehouse.sources[agent_source].coordinates
        waits_at_source = []
        scheduled = False

        while not scheduled:
            agent_source_busy = time in agent_scheduling_by_source[agent_source]
            if not agent_source_busy:
                for _ in range(ROUTE_SAMPLING_TRIES):
                    sampled_solution = sample_routing_request_route_from_database(warehouse, [request])
                    agent_plan = waits_at_source + sampled_solution

                    if not is_agent_plan_conflicting_with_plan(plan, agent_plan):
                        plan.append(agent_plan)
                        agent_scheduling_by_source[agent_source].add(time)
                        scheduled = True
                        break

                if not scheduled:
                    waits_at_source.append(agent_source_coordinates)
                    time += 1
            else:
                waits_at_source.append(agent_source_coordinates)
                time += 1
            # for _ in range(ROUTE_SAMPLING_TRIES):
            #     while not scheduled:
            #         agent_source_busy = time in agent_scheduling_by_source[agent_source]
            #         if not agent_source_busy:
            #             sampled_solution = sample_routing_request_route_from_database(warehouse, [request])
            #             agent_plan = waits_at_source + sampled_solution
            #
            #             if not is_agent_plan_conflicting_with_plan(plan, agent_plan):
            #                 plan.append(agent_plan)
            #                 agent_scheduling_by_source[agent_source].add(time)
            #                 print("found at time", time)
            #                 scheduled = True
            #         else:
            #             waits_at_source.append(agent_source_coordinates)
            #             time += 1
            #
            # if not scheduled:
            #     waits_at_source.append(agent_source_coordinates)
            #     time += 1
            #     print("trying again for time", time)

    return plan
