import random
import pandas as pd


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