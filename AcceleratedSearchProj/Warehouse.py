import sys
from math import floor
from queue import Queue
from typing import List, Set, Tuple
import matplotlib.pyplot as plt
from math import sqrt

PLOT_OBSTACLE_INTERIOR = True

SOURCE_OFFSET = 3
DESTINATION_OFFSET = 5
ALLOW_DIAGONAL_MOVEMENT = False
PRIORITIZE_AGENTS_WAITING_AT_SOURCE = True


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

