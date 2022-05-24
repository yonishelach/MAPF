import csv
import heapq
import math
import os
import random

import pandas as pd

import NoDeviationFactorDatabase
from EnvironmentUtils import count_plan_conflicts
from Utils import distance

MAX_APPROXIMATION = 0.04
MAX_CONTINUED_MOTION = 5
MAX_CONTINUED_MOTION_FACTOR = 4
MIDPOINT_DISTANCE_FROM_EDGES_FACTOR = 0.2
MAX_DEVIATION_FACTOR_IN_DATABASE = 1.7


def generate_midpoint_restricted_path_lengths_csv(warehouse, source_id, destination_id, data):
    warehouse_id = warehouse.warehouse_id

    target_dir = f'./csv_files/warehouse_{warehouse_id}/midpoint_restricted_path_lengths/'

    if not os.path.isdir(target_dir):
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    file_name = f'./csv_files/warehouse_{warehouse_id}/midpoint_restricted_path_lengths/from_source_{source_id}' \
                f'_to_destination_{destination_id}.csv'
    df = pd.DataFrame(data, columns=['deviation_factor', 'midpoint']).set_index('deviation_factor')
    df.to_csv(file_name)


def get_ideal_midpoint_restricted_path_length(source, destination, midpoint):
    midpoint_source_distance = midpoint.source_distance[source.source_id]
    midpoint_destination_distance = midpoint.destination_distance[destination.destination_id]
    return midpoint_source_distance + midpoint_destination_distance


def get_midpoint_restricted_path_lengths(warehouse, source, destination):
    ideal_path_length = source.destination_distance[destination.destination_id]
    vertices_heap = []

    for i, row in enumerate(warehouse.vertices):
        is_row_too_close_to_source = (i >= (warehouse.length * (1 - MIDPOINT_DISTANCE_FROM_EDGES_FACTOR)))
        is_row_too_close_to_destination = (i <= (warehouse.length * MIDPOINT_DISTANCE_FROM_EDGES_FACTOR))
        if is_row_too_close_to_source or is_row_too_close_to_destination:
            continue

        for vertex in row:
            if vertex.is_static_obstacle or vertex.destination_id != -1 or \
                    (vertex.source_id != -1 and vertex != source):
                continue

            ideal_mrp_length = get_ideal_midpoint_restricted_path_length(source, destination, vertex)
            deviation_factor = round(ideal_mrp_length / ideal_path_length, 2)
            if deviation_factor <= MAX_DEVIATION_FACTOR_IN_DATABASE:
                heapq.heappush(vertices_heap, (deviation_factor, vertex.coordinates))

    sorted_vertices = []
    while vertices_heap:
        sorted_vertices.append(heapq.heappop(vertices_heap))

    return sorted_vertices


def build_midpoint_restricted_database(warehouse, source, destination):
    NoDeviationFactorDatabase.create_warehouse_dir_if_does_not_exist(warehouse)

    midpoint_restricted_path_lengths = get_midpoint_restricted_path_lengths(warehouse, source, destination)
    generate_midpoint_restricted_path_lengths_csv(warehouse, source.source_id, destination.destination_id,
                                                  midpoint_restricted_path_lengths)


def get_midpoint_dataframe(warehouse, source_id, destination_id):
    warehouse_id = warehouse.warehouse_id
    file_path = f'./csv_files/warehouse_{warehouse_id}/midpoint_restricted_path_lengths/from_source_{source_id}' \
                f'_to_destination_{destination_id}.csv'
    file_exists = os.path.isfile(file_path)
    if not file_exists:
        print("Needed files not found. Please run initialize_database_preliminary_files(warehouse)")
        return None

    midpoint_df = pd.read_csv(file_path)
    return midpoint_df


def get_max_accuracy_for_deviation_factor(warehouse, source_id, destination_id, deviation_factor):
    midpoint_df = get_midpoint_dataframe(warehouse, source_id, destination_id)
    values_below_deviation_factor = midpoint_df.loc[midpoint_df.deviation_factor <= deviation_factor]
    min_value_below_deviation_factor = values_below_deviation_factor.iloc[-1]['deviation_factor']

    min_value_above_deviation_factor = 0
    values_above_deviation_factor = midpoint_df.loc[midpoint_df.deviation_factor >= deviation_factor]
    if values_above_deviation_factor.size != 0:
        min_value_above_deviation_factor = values_above_deviation_factor.iloc[0]['deviation_factor']

    deviation_factor_value_below_difference = round(abs(deviation_factor - min_value_below_deviation_factor), 2)
    deviation_factor_value_above_difference = round(abs(min_value_above_deviation_factor - deviation_factor), 2)

    is_deviation_below_too_large = (deviation_factor_value_below_difference >= MAX_APPROXIMATION)
    is_deviation_above_too_large = (deviation_factor_value_above_difference >= MAX_APPROXIMATION)

    if is_deviation_below_too_large and is_deviation_above_too_large:
        min_deviation_possible = min(deviation_factor_value_below_difference, deviation_factor_value_above_difference)
        print(f"Algorithm cannot supply good enough approximation for source_id={source_id}, "
              f"destination_id={destination_id}, devination_factor={deviation_factor}.\n"
              f"Can only approximate below {MAX_APPROXIMATION}, but the minimum approximation available is {min_deviation_possible}.\n")
        return -1

    if is_deviation_below_too_large:
        deviation_factor_value_below_difference = 0

    if is_deviation_above_too_large:
        deviation_factor_value_above_difference = 0

    max_accuracy = round(max(deviation_factor_value_below_difference, deviation_factor_value_above_difference), 2)

    return max_accuracy


def sample_midpoint_from_database(warehouse, source_id, destination_id, deviation_factor):
    midpoint_df = get_midpoint_dataframe(warehouse, source_id, destination_id)
    accuracy = get_max_accuracy_for_deviation_factor(warehouse, source_id,
                                                     destination_id, deviation_factor)
    if accuracy == -1:
        return -1

    relevant_data = midpoint_df.loc[round(abs(deviation_factor - midpoint_df.deviation_factor), 2) <= accuracy]

    sampled_midpoint = eval(relevant_data.midpoint.sample().item())
    return sampled_midpoint


def greedily_generate_path_from_midpoint_to_destination(warehouse, destination_id, midpoint_coordinates):
    destination = warehouse.destinations[destination_id]

    vertex = warehouse.vertices[midpoint_coordinates[0]][midpoint_coordinates[1]]
    path = [vertex.coordinates]

    while vertex != destination:
        current_destination_distance = vertex.destination_distance[destination_id] + 1

        improving_neighbors = [neighbor for neighbor in vertex.neighbors if
                               neighbor.destination_distance[destination_id] < current_destination_distance]
        if len(improving_neighbors) == 1:
            vertex = improving_neighbors[0]
        else:
            current_width_distance = abs(vertex.coordinates[0] - destination.coordinates[0])
            first_neighbor_width_distance = abs(improving_neighbors[0].coordinates[0] - destination.coordinates[0])
            is_first_neighbor_width_improver = first_neighbor_width_distance < current_width_distance
            width_improver = improving_neighbors[0] if is_first_neighbor_width_improver else improving_neighbors[1]
            length_improver = improving_neighbors[1] if is_first_neighbor_width_improver else improving_neighbors[0]

            current_length_distance = abs(vertex.coordinates[1] - destination.coordinates[1])
            vertex = random.choices([width_improver, length_improver],
                                    weights=[current_width_distance, 2*current_length_distance])[0]

        path.append(vertex.coordinates)

    return path


def greedily_generate_path_from_source_to_midpoint(warehouse, source_id, midpoint_coordinates):
    source = warehouse.sources[source_id]
    source_exit = list(source.neighbors)[0]

    vertex = warehouse.vertices[midpoint_coordinates[0]][midpoint_coordinates[1]]
    path = [vertex.coordinates]

    while vertex != source_exit:
        current_source_distance = vertex.source_distance[source_id] + 1

        improving_neighbors = [neighbor for neighbor in vertex.neighbors if
                               neighbor.source_distance[source_id] < current_source_distance]
        if len(improving_neighbors) == 1:
            vertex = improving_neighbors[0]
        else:
            current_width_distance = abs(vertex.coordinates[0] - source.coordinates[0])
            first_neighbor_width_distance = abs(improving_neighbors[0].coordinates[0] - source.coordinates[0])
            is_first_neighbor_width_improver = first_neighbor_width_distance < current_width_distance
            width_improver = improving_neighbors[0] if is_first_neighbor_width_improver else improving_neighbors[1]
            length_improver = improving_neighbors[1] if is_first_neighbor_width_improver else improving_neighbors[0]

            current_length_distance = abs(vertex.coordinates[1] - source.coordinates[1])
            vertex = random.choices([width_improver, length_improver],
                                    weights=[current_width_distance, 2*current_length_distance])[0]

        path.append(vertex.coordinates)

    path.append(source.coordinates)
    path.reverse()
    return path


# def generate_midpoint_restricted_path(warehouse, source_id, destination_id, deviation_factor):
#     midpoint = sample_midpoint_from_database(warehouse, source_id, destination_id, deviation_factor)
#     if midpoint == -1:
#         return []
#
#     path_to_midpoint = greedily_generate_path_from_source_to_midpoint(warehouse, source_id, midpoint)
#     path_from_midpoint = greedily_generate_path_from_midpoint_to_destination(warehouse, destination_id, midpoint)
#     path = path_to_midpoint + path_from_midpoint[1:]
#
#     return path


def generate_deviation_factor_matching_path_via_stochastic_search_with_reference_point(warehouse, source_id,
                                                                                       destination_id,
                                                                                       deviation_factor,
                                                                                       reference_point_coordinates=None,
                                                                                       continued_motion_factor=2):
    if not reference_point_coordinates:
        reference_point_coordinates = warehouse.destinations[destination_id].coordinates

    vertex = warehouse.sources[source_id]
    ideal_path_length = vertex.destination_distance[destination_id]
    target_path_length = ideal_path_length * deviation_factor

    path = [vertex.coordinates]
    current_path_length = len(path) + vertex.destination_distance[destination_id]
    destination = warehouse.destinations[destination_id]
    worsening_step = False
    step_direction = (1, 0)
    retry = False
    continued_motion_steps_taken = 0
    while abs(current_path_length - target_path_length) >= 2 and vertex != destination:
        if worsening_step and continued_motion_steps_taken < MAX_CONTINUED_MOTION:
            continued_motion_x_coordinates = vertex.coordinates[0] - step_direction[0]
            continued_motion_y_coordinates = vertex.coordinates[1] - step_direction[1]
            if warehouse.is_valid_vertex(continued_motion_x_coordinates, continued_motion_y_coordinates):
                continued_motion_vertex = warehouse.vertices[continued_motion_x_coordinates][
                    continued_motion_y_coordinates]
                if continued_motion_vertex.source_id == -1 and continued_motion_vertex.destination_id == -1 \
                        and random.choice([True for _ in range(continued_motion_factor)] + [False]):
                    vertex = continued_motion_vertex
                    path.append(vertex.coordinates)
                    current_path_length = len(path) + vertex.destination_distance[destination_id]
                    continued_motion_steps_taken += 1
                    continue
        else:
            continued_motion_steps_taken = 0

        neighbors = list(vertex.neighbors)
        neighbors_weights = []
        destination_to_remove = []
        non_zero_weight_neighbor_found = False
        for i, neighbor in enumerate(neighbors):
            if neighbor.destination_id != -1:
                destination_to_remove.append(i)
                retry = True
                break
            else:
                neighbor_step_direction = (vertex.coordinates[0] - neighbor.coordinates[0],
                                           vertex.coordinates[1] - neighbor.coordinates[1])
                is_oscillation = (neighbor_step_direction[0] + step_direction[0],
                                  neighbor_step_direction[1] + step_direction[1]) == (0, 0)
                is_step_upwards = (neighbor_step_direction == (-1, 0))
                if is_oscillation or is_step_upwards:
                    neighbors_weights.append(0)
                else:
                    vertex_weight = distance(vertex.coordinates, reference_point_coordinates) + 2
                    neighbors_weights.append(vertex_weight)
                    non_zero_weight_neighbor_found = True

        if destination_to_remove or (not non_zero_weight_neighbor_found):
            retry = True
            break

        chosen_neighbor = random.choices(neighbors, neighbors_weights)[0]
        step_direction = (vertex.coordinates[0] - chosen_neighbor.coordinates[0],
                          vertex.coordinates[1] - chosen_neighbor.coordinates[1])

        if vertex.destination_distance[destination_id] <= chosen_neighbor.destination_distance[destination_id]:
            worsening_step = True

        vertex = chosen_neighbor
        path.append(vertex.coordinates)
        current_path_length = len(path) + vertex.destination_distance[destination_id]

    if retry:
        if continued_motion_factor < MAX_CONTINUED_MOTION_FACTOR:
            continued_motion_factor += random.choice([True, False])

        return generate_deviation_factor_matching_path_via_stochastic_search_with_reference_point(warehouse, source_id,
                                                                                                  destination_id,
                                                                                                  deviation_factor,
                                                                                                  reference_point_coordinates,
                                                                                                  continued_motion_factor)

    path_from_midpoint = greedily_generate_path_from_midpoint_to_destination(warehouse, destination_id,
                                                                             vertex.coordinates)
    path += path_from_midpoint[1:]
    return path


def get_a_reference_point_far_from_source(warehouse, source_id, destination_id, deviation_factor):
    midpoint_df = get_midpoint_dataframe(warehouse, source_id, destination_id)
    values_above_deviation_factor = midpoint_df.loc[midpoint_df.deviation_factor >= deviation_factor]
    if values_above_deviation_factor.size == 0:
        highest_deviation_factor = midpoint_df.iloc[-1]['deviation_factor']
        highest_deviation_factor_options = midpoint_df.loc[midpoint_df.deviation_factor == highest_deviation_factor]

        sampled_midpoint = eval(highest_deviation_factor_options.midpoint.sample().item())
    else:
        sampled_midpoint = eval(values_above_deviation_factor.midpoint.sample().item())

    return sampled_midpoint


def generate_approximately_matching_deviation_factor_path_via_midpoint(warehouse, source_id, destination_id,
                                                                       deviation_factor):
    midpoint = sample_midpoint_from_database(warehouse, source_id, destination_id, deviation_factor)
    if midpoint == -1:
        return []

    path_to_midpoint = greedily_generate_path_from_source_to_midpoint(warehouse, source_id, midpoint)
    path_from_midpoint = greedily_generate_path_from_midpoint_to_destination(warehouse, destination_id, midpoint)
    path = path_to_midpoint + path_from_midpoint[1:]

    return path


def path_with_deviation_factor_header_file_field_names(ideal_path_length, deviation_factor):
    warehouse_fields = ['warehouse_id', 'source_id', 'destination_id']
    possible_path_timestamps = [f'time={i}' for i in range(round(ideal_path_length * (deviation_factor + 0.1)))]

    return warehouse_fields + possible_path_timestamps


def create_path_file_with_deviation_factor_if_does_not_exist(file_path, field_names):
    file_exists = os.path.isfile(file_path)
    if not file_exists:
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()


def export_source_destination_deviation_factor_specific_plan_to_csv(warehouse, source_id, destination_id,
                                                                    deviation_factor, plan):
    warehouse_id = warehouse.warehouse_id

    dir_path = f"./csv_files/warehouse_{warehouse_id}/paths/from_source_{source_id}_to_destination_{destination_id}/"
    if not os.path.isdir(dir_path):
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)

    ideal_path_length = warehouse.sources[source_id].destination_distance[destination_id]
    field_names = path_with_deviation_factor_header_file_field_names(ideal_path_length, deviation_factor)

    file_path = dir_path + f'deviation_factor_{deviation_factor}.csv'
    create_path_file_with_deviation_factor_if_does_not_exist(file_path, field_names)

    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)

        for path in plan:
            path_info = {'warehouse_id': warehouse.warehouse_id, 'source_id': source_id,
                         'destination_id': destination_id}
            for i, coordinates in enumerate(path):
                path_info[f'time={i}'] = coordinates

            writer.writerow(path_info)


def generate_deviation_factor_matching_path_via_stochastic_search(warehouse, source_id, destination_id,
                                                                  deviation_factor):
    reference_point = get_a_reference_point_far_from_source(warehouse, source_id, destination_id, deviation_factor)
    return generate_deviation_factor_matching_path_via_stochastic_search_with_reference_point(warehouse, source_id,
                                                                                              destination_id,
                                                                                              deviation_factor,
                                                                                              reference_point)


def sample_routing_request_path_from_database(warehouse, routing_request, deviation_factor):
    warehouse_id = warehouse.warehouse_id
    source_id, destination_id = routing_request[0], routing_request[1]
    file_path = f"./csv_files/warehouse_{warehouse_id}/paths/from_source_{source_id}_to_destination_{destination_id}/" \
                f"deviation_factor_{deviation_factor}.csv"

    file_exists = os.path.isfile(file_path)
    if not file_exists:
        print("File does not exist:" + file_path)
        return []

    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        file_content = list(reader)
        file_content_without_header = file_content[1:]
        sampled_row_in_string_format = random.choice(file_content_without_header)
        sampled_path_in_string_format = sampled_row_in_string_format[3:]
        path = [eval(coordinates) for coordinates in sampled_path_in_string_format if coordinates]
        return path


##################################################################
#   TODO: keep only the functions below this line in this file   #
##################################################################
# def sample_path_database(warehouse, routing_requests, deviation_factors, appear_at_source_times):
#     """
#     Returns a plan, where plan[i] is a path from routing_requests[i][0] to routing_requests[i][0],
#         with a deviation_factor of deviation_factors[i] and an appear_at_source[i] timestamps of
#         wait steps at source performed by the agent.
#
#     routing_requests: List of tuples of the format (source_id, destination_id). i.e. List(Tuple(int, int))
#
#     deviation_factors: List of deviation factors relating to the routing_requests. i.e. List(float).
#         deviation_factors[i] dictates the deviation_factor retrieved for routing_requests[i].
#
#     appear_at_source_time: Lists of the last timestamp before the agent leaves her source. i.e. List(int)
#     """
#     if (len(routing_requests) != len(deviation_factors)) or (len(routing_requests) != len(appear_at_source_times)):
#         print("Please supply inputs such that 'routing_requests', 'deviation_factors' and 'appear_at_source_time' "
#               "all have the same length")
#         return
#
#     plan = []
#     for i, routing_request in enumerate(routing_requests):
#         deviation_factor = deviation_factors[i]
#         sampled_path = sample_routing_request_path_from_database(warehouse, routing_request, deviation_factor)
#
#         print(sampled_path)
#         appear_at_source_time = appear_at_source_times[i]
#         waits_at_source = [sampled_path[0] for _ in range(appear_at_source_time)]
#         path_with_waits_at_source = waits_at_source + sampled_path
#         plan.append(path_with_waits_at_source)
#         print(path_with_waits_at_source)
#
#     return plan


def sample_path_database(warehouse, source_id, destination_id, deviation_factor, number_of_paths=30):
    """
    Returns a list of length number_of_paths.
        Each element in the list is a path (a list of coordinates), leading from warehouse.sources[source_id] to
        warehouse.destinations[destination_id]. Each of these paths has the given deviation_factor, and is queried from
        the database file in path: "./csv_files/warehouse_{warehouse_id}/paths/
                                      from_source_{source_id}_to_destination_{destination_id}/
                                      deviation_factor_{deviation_factor}.csv"

        If not enough paths exist in the database file above, a message is printed, and all paths in the database file
        are returned instead.
    """
    warehouse_id = warehouse.warehouse_id
    plan = []

    file_path = f"./csv_files/warehouse_{warehouse_id}/paths/from_source_{source_id}_to_destination_{destination_id}/" \
                f"deviation_factor_{deviation_factor}.csv"

    file_exists = os.path.isfile(file_path)
    if not file_exists:
        print("File does not exist:" + file_path)
        return []

    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        file_content = list(reader)
        file_content_without_header = file_content[1:]

        number_of_paths_available_in_file = len(file_content_without_header)
        if number_of_paths_available_in_file < number_of_paths:
            print(f"Trying to sample {number_of_paths} paths, but only {number_of_paths_available_in_file} are "
                  f"currently available in file.")
            print(f"number_of_paths reduces to {number_of_paths_available_in_file}")
            number_of_paths = number_of_paths_available_in_file

        sampled_rows_in_string_format = random.sample(file_content_without_header, number_of_paths)
        for sampled_row_in_string_format in sampled_rows_in_string_format:
            sampled_path_in_string_format = sampled_row_in_string_format[3:]
            path = [eval(coordinates) for coordinates in sampled_path_in_string_format if coordinates]
            plan.append(path)

    return plan


def generate_path_database(warehouse, paths_per_deviation_factor=30, deviation_factors=None,
                           path_generation_algorithm=generate_deviation_factor_matching_path_via_stochastic_search):
    """
    Generates a path database for the given parameters via the path_generation_algorithm.
    If files sharing the database file names exist, the newly generated data is concatenated to them.
        Otherwise, the function creates the files and writes the newly generated data to them.

    Returns a plan with the generated paths.

    deviation_factors: If no value is supplied, a default value of [1.01, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7] is used.
    path_generation_algorithm: Supports values in [generate_deviation_factor_matching_path_via_stochastic_search,
                                                   generate_approximately_matching_deviation_factor_path_via_midpoint].
    """
    if deviation_factors is None:
        deviation_factors = [1.01, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]

    plan = []
    source_destination_deviation_factor_specific_plan = []
    for source_id in range(warehouse.number_of_sources):
        for destination_id in range(warehouse.number_of_destinations):
            for deviation_factor in deviation_factors:
                for _ in range(paths_per_deviation_factor):
                    path = path_generation_algorithm(warehouse, source_id, destination_id, deviation_factor)

                    if path:
                        source_destination_deviation_factor_specific_plan.append(path)
                    else:
                        break
                export_source_destination_deviation_factor_specific_plan_to_csv(warehouse, source_id, destination_id,
                                                                                deviation_factor,
                                                                                source_destination_deviation_factor_specific_plan)
                plan += source_destination_deviation_factor_specific_plan
                source_destination_deviation_factor_specific_plan = []
    return plan


def initialize_database_preliminary_files(warehouse):
    """
    Creates the warehouse folder, containing the warehouse layout .csv and .png files.
    Creates the midpoint restricted database.

    Note: This function overrides existing files.
    Note: This function does not build the path database.
    """
    notification_rate = 10
    print(f"Initializing warehouse directory for warehouse {warehouse.warehouse_id}")
    NoDeviationFactorDatabase.create_warehouse_dir(warehouse)
    print("Done")
    print("***")

    print("Initializing midpoint restricted database")
    for source in warehouse.sources:
        source_id = source.source_id
        if source_id % notification_rate == 0:
            print(
                f"Building midpoint restricted database for sources {source_id} - {source_id + (notification_rate - 1)}")

        for destination in warehouse.destinations:
            build_midpoint_restricted_database(warehouse, source, destination)
    print("Done")
    print("***")
