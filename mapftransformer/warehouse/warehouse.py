# This code is taken from past semesters,so it is not documented. With this code we build our custom warehouse object.
import csv
import heapq
import sys
import os
from math import floor, sqrt
from queue import Queue
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd

PLOT_OBSTACLE_INTERIOR = True

SOURCE_OFFSET = 3
DESTINATION_OFFSET = 5
ALLOW_DIAGONAL_MOVEMENT = False
PRIORITIZE_AGENTS_WAITING_AT_SOURCE = True

MIDPOINT_DISTANCE_FROM_EDGES_FACTOR = 0.2
MAX_DEVIATION_FACTOR_IN_DATABASE = 1.7


class Warehouse:
    class WarehouseNode:
        def __init__(self, coordinates, number_of_sources, number_of_destinations):
            self.coordinates = coordinates
            self.source_distance = [0 for _ in range(number_of_sources)]
            # self.source_routes = [[] for _ in range(number_of_sources)]
            self.destination_distance = [0 for _ in range(number_of_destinations)]
            # self.destination_routes = [[] for _ in range(number_of_destinations)]
            self.is_static_obstacle = False
            self.source_id = -1  # not optimal solution, but simplifies implementation
            self.destination_id = -1  # not optimal solution, but simplifies implementation
            self.neighbors = set()
            self.routing_requests = set()

        def get_destination_distance(self, destination_id):
            return self.destination_distance[destination_id]

    """
         Uses a breadth first search to calculate the distances of all vertices in the warehouse from the ith 
         destination.
         This algorithm avoids static obstacles and ignores dynamic obstacles
    """

    def set_ith_destination_distances(self, i):
        destination = self.destinations[i]
        destination_coordinates = destination.coordinates
        destination_entrance = self.vertices[destination_coordinates[0] + 1][destination_coordinates[1]]

        destination.destination_distance[i] = 0
        destination_entrance.destination_distance[i] = 1
        # destination_entrance.destination_routes[i].append(destination_coordinates)

        queue = Queue()
        queue.put(destination_entrance)
        visited = set()
        visited.add(destination)
        visited.add(destination_entrance)
        while not queue.empty():
            u = queue.get()
            for v in u.neighbors:
                if v not in visited:
                    v.destination_distance[i] = u.destination_distance[i] + 1
                    # v.destination_routes[i].append(u.coordinates)
                    # for node in u.destination_routes[i]:
                    #     v.destination_routes[i].append(node)
                    visited.add(v)
                    queue.put(v)
            visited.add(u)
        for source in self.sources:
            source_coordinates = source.coordinates
            source_entrance = self.vertices[source_coordinates[0] - 1][source_coordinates[1]]
            source.destination_distance[i] = 1 + source_entrance.destination_distance[i]
            # source.destination_routes[i].append(source_entrance.coordinates)
            # for node in source_entrance.destination_routes[i]:
            #     source.destination_routes[i].append(node)

    def set_destination_distances(self):
        for i in range(self.number_of_destinations):
            self.set_ith_destination_distances(i)

    """
         Uses a breadth first search to calculate the distances of all vertices in the warehouse from the ith 
         source.
         This algorithm avoids static obstacles and ignores dynamic obstacles
    """

    def set_ith_source_distances(self, i):
        source = self.sources[i]
        source_coordinates = source.coordinates

        source.source_distance[i] = 0
        # source.source_routes[i].append(source_coordinates)

        queue = Queue()
        queue.put(source)
        visited = set()
        visited.add(source)
        while not queue.empty():
            u = queue.get()
            for v in u.neighbors:
                if v not in visited:
                    v.source_distance[i] = u.source_distance[i] + 1
                    # v.source_routes[i].append(u.coordinates)
                    # for node in u.source_routes[i]:
                    #     v.source_routes[i].append(node)
                    visited.add(v)
                    queue.put(v)
            visited.add(u)

        for other_source in self.sources:
            if other_source == source:
                continue

            other_source.source_distance[i] = sys.maxsize

    def set_source_distances(self):
        for i in range(self.number_of_sources):
            self.set_ith_source_distances(i)

    def adjust_destinations_neighbors(self):
        for destination in self.destinations:
            destination_coordinates = destination.coordinates

            for neighbor in destination.neighbors:
                if (neighbor.coordinates[0] - destination_coordinates[0],
                    neighbor.coordinates[1] - destination_coordinates[1]) != (1, 0):
                    neighbor.neighbors.remove(destination)

            destination.neighbors = set()

    def adjust_sources_neighbors(self):
        for source in self.sources:
            source_coordinates = source.coordinates
            neighbors_to_remove = []

            for neighbor in source.neighbors:
                neighbor.neighbors.remove(source)

                neighbor_coordinates = neighbor.coordinates
                if (source_coordinates[0] - neighbor_coordinates[0], source_coordinates[1] - neighbor_coordinates[1]) \
                        != (1, 0):
                    neighbors_to_remove.append(neighbor)

            source.neighbors = source.neighbors.difference(neighbors_to_remove)

    def set_sources_and_destinations(self, number_of_targets, row_idx, target_array, is_destination=False):
        targets_with_dummies = number_of_targets
        distance_between_targets = floor(self.width / targets_with_dummies)

        first_target_position = 0
        last_dummy_position = distance_between_targets * targets_with_dummies

        for i, column_idx in enumerate(range(first_target_position, last_dummy_position, distance_between_targets)):
            vertex = self.vertices[row_idx][column_idx]
            target_array.append(vertex)

            if is_destination:
                vertex.destination_id = i
            else:
                vertex.source_id = i

    def set_destinations(self):
        self.set_sources_and_destinations(self.number_of_destinations, 0, self.destinations, True)

    def set_sources(self):
        self.set_sources_and_destinations(self.number_of_sources, self.length - 1, self.sources)

    def is_valid_vertex(self, row_idx, column_idx):
        if (0 <= row_idx < self.length) and (0 <= column_idx < self.width):
            return not self.vertices[row_idx][column_idx].is_static_obstacle

        return False

    """
    Note:   Diagonal neighbors are valid neighbors, while static obstacles are not.
            Also, a vertex cannot be a neighbor of itself
    """

    def set_neighbors(self):
        for row in self.vertices:
            for vertex in row:
                if vertex.is_static_obstacle:
                    continue

                for i in [-1, 0, 1]:
                    row_idx = vertex.coordinates[0] + i

                    for j in [-1, 0, 1]:
                        if not ALLOW_DIAGONAL_MOVEMENT and i ** 2 + j ** 2 != 1:
                            continue

                        column_idx = vertex.coordinates[1] + j
                        if (i != 0 or j != 0) and self.is_valid_vertex(row_idx, column_idx):
                            neighbor = self.vertices[row_idx][column_idx]
                            vertex.neighbors.add(neighbor)

    """
        Sets a rectangular static obstacle with a corner at the given indices
    """

    def set_static_obstacle(self, row_idx, column_idx):
        obstacle_corners = set()
        for i in range(self.static_obstacle_length):
            obstacle_row_idx = row_idx + i

            if 0 <= obstacle_row_idx < self.length:
                for j in range(self.static_obstacle_width):
                    obstacle_column_idx = column_idx + j

                    if 0 <= obstacle_column_idx < self.width:
                        self.vertices[obstacle_row_idx][obstacle_column_idx].is_static_obstacle = True
                        self.static_obstacles.add((obstacle_row_idx, obstacle_column_idx))

                        # used for animations
                        # if i == 0 or i == self.static_obstacle_length - 1 or j == 0 \
                        #         or j == self.static_obstacle_width - 1:
                        #     self.static_obstacle_corners.add((obstacle_row_idx, obstacle_column_idx))
                        #     obstacle_corners.add((obstacle_row_idx, obstacle_column_idx))
        # self.static_obstacle_coordinates_split_by_obstacle.append(obstacle_corners)

    def set_static_obstacles(self):
        # corners_coordinates = WAREHOUSE_CORNERS[self.warehouse_id - 1]
        for corner_coordinates in self.static_obstacle_layout:
            self.set_static_obstacle(corner_coordinates[0], corner_coordinates[1])

    def initialize_vertices(self):
        for row_idx in range(self.length):
            column = []

            for column_idx in range(self.width):
                coordinates = (row_idx, column_idx)
                new_vertex = self.WarehouseNode(coordinates, self.number_of_sources, self.number_of_destinations)

                column.append(new_vertex)

            self.vertices.append(column)

    def set_mid_points(self):
        for source in self.sources:
            mid_points = []
            for i, distance in enumerate(source.destination_distance):
                mid_points.append(source.destination_routes[i][distance // 2])
            self.sources_to_destinations_mid_point.append(mid_points)

    def set_averages(self):
        for j, source in enumerate(self.sources):
            averages_to_mid_point = []
            for i, route in enumerate(source.destination_routes):
                euclidian_distance_to_mid_point = 0.0
                mid_point_coordinates = self.sources_to_destinations_mid_point[j][i]
                for node_coordinates in route:
                    euclidian_distance_to_mid_point += sqrt(pow(node_coordinates[0] - mid_point_coordinates[0], 2) +
                                                            pow(node_coordinates[1] - mid_point_coordinates[1], 2))
                averages_to_mid_point.append(euclidian_distance_to_mid_point / source.destination_distance[i])
            self.sources_to_destinations_average_euclidian_distance_to_mid_point.append(averages_to_mid_point)

    # def print(self):
    #     for i, source in enumerate(self.sources):
    #         print("source.destination_distance:", source.destination_distance)
    #         print("source.destination_routes:", source.destination_routes)
    #         print("self.sources_to_destinations_mid_point:", self.sources_to_destinations_mid_point[i])
    #         print("self.sources_to_destinations_average_euclidian_distance_to_mid_point", self.sources_to_destinations_average_euclidian_distance_to_mid_point[i])

    def __init__(self, warehouse_id, length, width, number_of_sources, number_of_destinations, static_obstacle_length,
                 static_obstacle_width, static_obstacle_layout, is_warehouse_searchable):
        self.warehouse_id = warehouse_id
        self.length = length
        self.width = width
        self.number_of_sources = number_of_sources
        self.number_of_destinations = number_of_destinations
        self.static_obstacle_length = static_obstacle_length
        self.static_obstacle_width = static_obstacle_width
        self.static_obstacle_layout = static_obstacle_layout

        self.vertices: List[List[Warehouse.WarehouseNode]] = []
        self.static_obstacles = set()
        # self.static_obstacle_corners: Set[Tuple[int, int]] = set()
        # self.static_obstacle_coordinates_split_by_obstacle = []
        self.sources: List[Warehouse.WarehouseNode] = []
        self.destinations: List[Warehouse.WarehouseNode] = []
        # self.sources_to_destinations_mid_point: List[List[Tuple[int, int]]] = []
        # self.sources_to_destinations_average_euclidian_distance_to_mid_point: List[List[float]] = []

        self.initialize_vertices()
        self.set_static_obstacles()
        self.set_neighbors()

        self.set_sources()
        self.set_destinations()

        self.adjust_sources_neighbors()
        self.adjust_destinations_neighbors()
        if is_warehouse_searchable:
            self.set_source_distances()
            self.set_destination_distances()
        # self.set_mid_points()
        # self.set_averages()
        # self.print()

    def initialize_database_preliminary_files(self):
        """
        Creates the warehouse folder, containing the warehouse layout .csv and .png files.
        Creates the midpoint restricted database.

        Note: This function overrides existing files.
        Note: This function does not build the path database.
        """
        notification_rate = 10
        print(f"Initializing warehouse directory for warehouse {self.warehouse_id}")
        create_warehouse_dir(self)
        print("Done")
        print("***")

        print("Initializing midpoint restricted database")
        for source in self.sources:
            source_id = source.source_id
            if source_id % notification_rate == 0:
                print(
                    f"Building midpoint restricted database for sources {source_id} - {source_id + (notification_rate - 1)}")

            for destination in self.destinations:
                build_midpoint_restricted_database(self, source, destination)
        print("Done")
        print("***")

    def plot_obstacles(self):
        for point in self.static_obstacle_layout:
            lower_left_corner = [point[0], point[1]]
            upper_left_corner = [point[0] + self.static_obstacle_length - 1, point[1]]
            upper_right_corner = [point[0] + self.static_obstacle_length - 1, point[1] + self.static_obstacle_width - 1]
            lower_right_corner = [point[0], point[1] + self.static_obstacle_width - 1]

            # draw corners
            corner_x_values = [lower_left_corner[0], upper_left_corner[0], upper_right_corner[0], lower_right_corner[0],
                               lower_left_corner[0]]
            corner_y_values = [lower_left_corner[1], upper_left_corner[1], upper_right_corner[1], lower_right_corner[1],
                               lower_left_corner[1]]
            plt.fill(corner_y_values, corner_x_values, c='#bababa', linewidth=2)
            plt.plot(corner_y_values, corner_x_values, c='#adadad', linewidth=2)
            #
            # # draw diagonals
            # primary_diagonal_x = [lower_right_corner[0], upper_left_corner[0]]
            # primary_diagonal_y = [lower_right_corner[1], upper_left_corner[1]]
            # plt.plot(primary_diagonal_y, primary_diagonal_x, c='gray', linewidth=1)
            #
            # secondary_diagonal_x = [lower_left_corner[0], upper_right_corner[0]]
            # secondary_diagonal_y = [lower_left_corner[1], upper_right_corner[1]]
            # plt.plot(secondary_diagonal_y, secondary_diagonal_x, c='gray', linewidth=1)

    def plot_layout(self, plot_grid=False):
        # plt.style.use('tableau-colorblind10')

        fig = plt.figure(figsize=(8, 7), dpi=100)
        ax = plt.axes(xlim=(0, self.width - 1), ylim=(0, self.length - 1))
        for source in self.sources:
            plt.scatter(source.coordinates[1], source.coordinates[0], s=250, marker='v', c='#00964b')

        self.plot_obstacles()

        for destination in self.destinations:
            plt.scatter(destination.coordinates[1], destination.coordinates[0], s=250, marker='^', c='hotpink')

        plt.xlim(-0.5, self.width - 0.5)
        plt.ylim(0, self.length - 1)
        if plot_grid:
            plt.grid()

        return fig, ax

    def show(self):
        self.plot_layout()
        plt.show()
        plt.clf()

    def save_image(self):
        self.plot_layout()
        plt.grid()
        plt.savefig(f'warehouse_{self.warehouse_id}_layout.png')


def create_warehouse_dir(warehouse):
    target_dir = "./csv_files/warehouse_{}/".format(warehouse.warehouse_id)

    if not os.path.isdir(target_dir):
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    export_warehouse_information_to_csv(warehouse)
    export_warehouse_layout_image(warehouse)


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


def export_warehouse_layout_image(warehouse):
    plot_grid = True
    warehouse.plot_layout(plot_grid)
    warehouse_id = warehouse.warehouse_id
    plt.savefig(f'./csv_files/warehouse_{warehouse_id}/warehouse_{warehouse_id}_layout.png')


def build_midpoint_restricted_database(warehouse, source, destination):
    create_warehouse_dir_if_does_not_exist(warehouse)

    midpoint_restricted_path_lengths = get_midpoint_restricted_path_lengths(warehouse, source, destination)
    generate_midpoint_restricted_path_lengths_csv(warehouse, source.source_id, destination.destination_id,
                                                  midpoint_restricted_path_lengths)


def create_warehouse_dir_if_does_not_exist(warehouse):
    target_dir = "./csv_files/warehouse_{}/".format(warehouse.warehouse_id)

    if not os.path.isdir(target_dir):
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        export_warehouse_information_to_csv(warehouse)
        export_warehouse_layout_image(warehouse)


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


def get_ideal_midpoint_restricted_path_length(source, destination, midpoint):
    midpoint_source_distance = midpoint.source_distance[source.source_id]
    midpoint_destination_distance = midpoint.destination_distance[destination.destination_id]
    return midpoint_source_distance + midpoint_destination_distance


def generate_midpoint_restricted_path_lengths_csv(warehouse, source_id, destination_id, data):
    warehouse_id = warehouse.warehouse_id

    target_dir = f'./csv_files/warehouse_{warehouse_id}/midpoint_restricted_path_lengths/'

    if not os.path.isdir(target_dir):
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    file_name = f'./csv_files/warehouse_{warehouse_id}/midpoint_restricted_path_lengths/from_source_{source_id}' \
                f'_to_destination_{destination_id}.csv'
    df = pd.DataFrame(data, columns=['deviation_factor', 'midpoint']).set_index('deviation_factor')
    df.to_csv(file_name)


# def generate_database(warehouse_id, max_number_of_agents, number_of_agents_incrementation_step=1,
#                       number_of_samples=1000):
#     # Our custom warehouse:
#     warehouse = Warehouse(
#         warehouse_id=1,
#         length=
#                           )
#     if warehouse_id == 100:
#         length = 40
#         width = 40
#         number_of_sources = 20
#         number_of_destinations = 10
#         obstacle_length = round(0.1 * length)
#         obstacle_width = round(0.1 * width)
#         obstacle_layout = []
#
#         return Warehouse(warehouse_id, length, width, number_of_sources, number_of_destinations, obstacle_length,
#                          obstacle_width, obstacle_layout, is_warehouse_searchable)
#     warehouse = generate_warehouse(warehouse_id)
#     initialize_database_preliminary_files(warehouse)
#
#     print("***")
#     print("running experiments")
#     run_experiments_to_generate_main_data_file(warehouse, max_number_of_agents, number_of_samples)
#     # for i in range(1, max_number_of_agents + 1, number_of_agents_incrementation_step):
#     #     run_experiments_to_generate_main_data_file(warehouse, i, number_of_samples)
#     #     print(f"experiment {i} complete")
#     print("***")
#     print("Done")
