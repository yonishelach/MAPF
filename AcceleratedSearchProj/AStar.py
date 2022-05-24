import heapq
import random
from sys import maxsize
from typing import Dict, Set, Tuple

import EnvironmentUtils
from RoutingRequest import ALLOW_DIAGONAL_MOVEMENT
from Utils import distance

PRIORITIZE_AGENTS_WAITING_AT_SOURCE = False
RANDOMIZE_NEIGHBORS = True


class AStar:
    class Node:
        def __init__(self, vertex, h_value, g_value, parent, is_source=False, waits_at_source=0, energy_cost=0):
            self.vertex = vertex
            self.h_value = h_value
            self.g_value = g_value
            self.parent = parent
            self.energy_cost = energy_cost

            self.f_value = h_value + g_value
            self.is_source = is_source
            self.waits_at_source = waits_at_source

        def __eq__(self, other):
            return self.vertex == other.vertex

        def __lt__(self, other):
            # if self.f_value == other.f_value:
            #     return self.waits_at_source > other.waits_at_source

            return self.f_value < other.f_value

        def update_info(self, possible_g_value, possible_parent):
            if possible_g_value < self.g_value:
                self.g_value = possible_g_value
                self.f_value = self.g_value + self.h_value

                self.parent = possible_parent
                return True

            return False

        def is_adjacent_vertex(self, vertex_coordinates):
            self_coordinates = self.vertex.coordinates
            adjacent_differences = {-1, 0, 1}

            return (self_coordinates[0] - vertex_coordinates[0] in adjacent_differences) and \
                   (self_coordinates[1] - vertex_coordinates[1] in adjacent_differences)

        def remove_intersecting_edges_collisions(self, warehouse, valid_neighbors, plan):
            current_coordinates = self.vertex.coordinates
            valid_neighbors_coordinates = [neighbor.coordinates for neighbor in valid_neighbors]
            valid_neighbors_edge_midpoints = [((current_coordinates[0] + coordinates[0]) / 2,
                                               (current_coordinates[1] + coordinates[1]) / 2)
                                              for coordinates in valid_neighbors_coordinates]

            for routing_request_plan in plan:
                if self.g_value + 1 < len(routing_request_plan):
                    routing_request_current_coordinates = routing_request_plan[self.g_value]
                    routing_request_next_coordinates = routing_request_plan[self.g_value + 1]

                    if not self.is_adjacent_vertex(routing_request_current_coordinates) or not \
                            self.is_adjacent_vertex(routing_request_next_coordinates):
                        continue

                    routing_request_current_coordinates = routing_request_plan[self.g_value]
                    routing_request_next_coordinates = routing_request_plan[self.g_value + 1]
                    routing_request_midpoint_coordinates = (
                        (routing_request_current_coordinates[0] + routing_request_next_coordinates[0]) / 2,
                        (routing_request_current_coordinates[1] + routing_request_next_coordinates[1]) / 2)

                    if routing_request_midpoint_coordinates in valid_neighbors_edge_midpoints:
                        conflicting_neighbor_index = valid_neighbors_edge_midpoints.index(
                            routing_request_midpoint_coordinates)
                        conflicting_neighbor_coordinates = valid_neighbors_coordinates[conflicting_neighbor_index]
                        conflicting_neighbor = warehouse.vertices[conflicting_neighbor_coordinates[0]][
                            conflicting_neighbor_coordinates[1]]
                        valid_neighbors.remove(conflicting_neighbor)
                        if not valid_neighbors:
                            return

                        valid_neighbors_coordinates.pop(conflicting_neighbor_index)
                        valid_neighbors_edge_midpoints.pop(conflicting_neighbor_index)

        def remove_edge_collisions(self, warehouse, valid_neighbors, plan):
            valid_neighbors_coordinates = {neighbor.coordinates for neighbor in valid_neighbors}
            current_self_coordinates = self.vertex.coordinates

            for routing_request_plan in plan:
                if self.g_value + 1 < len(routing_request_plan):
                    current_other_coordinates = routing_request_plan[self.g_value]
                    next_other_coordinates = routing_request_plan[self.g_value + 1]

                    if current_other_coordinates in valid_neighbors_coordinates and \
                            current_self_coordinates == next_other_coordinates:
                        valid_neighbors.remove(warehouse.vertices[current_other_coordinates[0]]
                                               [current_other_coordinates[1]])
                        if not valid_neighbors:
                            return
                        valid_neighbors_coordinates.remove(current_other_coordinates)

        def remove_vertex_collisions(self, warehouse, valid_neighbors, plan):
            valid_neighbors_coordinates = {neighbor.coordinates for neighbor in valid_neighbors}

            for routing_request_plan in plan:
                if self.g_value + 1 < len(routing_request_plan):
                    next_other_coordinates = routing_request_plan[self.g_value + 1]

                    if next_other_coordinates in valid_neighbors_coordinates:
                        valid_neighbors.remove(warehouse.vertices[next_other_coordinates[0]][next_other_coordinates[1]])
                        if not valid_neighbors:
                            return
                        valid_neighbors_coordinates.remove(next_other_coordinates)

        def remove_constrained_neighbors(self, warehouse, valid_neighbors,
                                         constraints: Dict[int, Set[Tuple[int, int]]]):
            # remove constricted vertexes
            valid_neighbors_coordinates = {neighbor.coordinates for neighbor in valid_neighbors}
            if (self.g_value + 1) in constraints:
                for coordinates in constraints[self.g_value + 1]:
                    if coordinates in valid_neighbors_coordinates:
                        valid_neighbors.remove(warehouse.vertices[coordinates[0]][coordinates[1]])

        def get_valid_neighbors_from_plan(self, warehouse, plan, constraints: Dict[int, Set[Tuple[int, int]]] = None):
            valid_neighbors = self.vertex.neighbors.copy()

            if constraints is not None:
                self.remove_constrained_neighbors(warehouse, valid_neighbors, constraints)

            self.remove_vertex_collisions(warehouse, valid_neighbors, plan)

            if ALLOW_DIAGONAL_MOVEMENT:
                self.remove_intersecting_edges_collisions(warehouse, valid_neighbors, plan)
            else:
                self.remove_edge_collisions(warehouse, valid_neighbors, plan)

            return valid_neighbors

        def get_valid_neighbors_from_added_obstacles(self, added_obstacles):
            valid_neighbors = self.vertex.neighbors.copy()
            colliding_neighbors = set()

            for neighbor in valid_neighbors:
                if neighbor.coordinates in added_obstacles:
                    colliding_neighbors.add(neighbor)

            valid_neighbors -= colliding_neighbors

            return valid_neighbors

    def __init__(self, source, destination):
        self.nodes = []
        self.source = source
        self.destination = destination

    def reconstruct_route(self):
        current_node = self.destination
        reversed_route = [current_node.vertex.coordinates]

        while current_node.parent:
            current_node = current_node.parent
            reversed_route.append(current_node.vertex.coordinates)

        return reversed_route[::-1]

    def space_time_search(self, warehouse, routing_request, plan, is_first_routing_request=False, wait_at_source_left=0,
                          constraints: Dict[int, Set[Tuple[int, int]]] = None):
        source = self.source
        destination = self.destination
        destination_id = routing_request.destination.destination_id

        priority_queue = []
        heapq.heappush(priority_queue, source)
        visited = {(source.g_value, source.vertex, source.energy_cost): source}

        while priority_queue:
            current_node = heapq.heappop(priority_queue)

            if current_node == destination:
                self.destination = current_node
                route = self.reconstruct_route()
                return route

            path_cost = current_node.g_value + 1

            valid_neighbors = current_node.get_valid_neighbors_from_plan(warehouse, plan, constraints)
            # if current_node.is_source:
            if current_node.vertex in warehouse.sources:
                valid_neighbors.add(current_node.vertex)
                if wait_at_source_left > 0:
                    valid_neighbors = [current_node.vertex]
                    wait_at_source_left -= 1

            for neighbor in valid_neighbors:
                is_neighbor_equals_source = neighbor == source.vertex
                neighbor_energy_cost = current_node.energy_cost + (not is_neighbor_equals_source)

                if not EnvironmentUtils.is_energy_cost_valid(warehouse, neighbor_energy_cost):
                    continue

                if (path_cost, neighbor, neighbor_energy_cost) in visited:
                    neighbor_node = visited[(path_cost, neighbor, neighbor_energy_cost)]

                    if path_cost < neighbor_node.g_value or \
                            (PRIORITIZE_AGENTS_WAITING_AT_SOURCE and path_cost == neighbor_node.g_value and
                             current_node.waits_at_source > neighbor_node.waits_at_source):
                        neighbor_node.parent = current_node
                        neighbor_node.g_value = path_cost
                        neighbor_node.f_value = neighbor_node.g_value + neighbor_node.h_value
                        neighbor_node.waits_at_source = current_node.waits_at_source + is_neighbor_equals_source
                        neighbor_node.energy_cost = neighbor_energy_cost

                        heapq.heappush(priority_queue, neighbor_node)

                else:
                    neighbor_node = AStar.Node(neighbor, neighbor.destination_distance[destination_id],
                                               path_cost, current_node, is_neighbor_equals_source,
                                               current_node.waits_at_source + is_neighbor_equals_source,
                                               neighbor_energy_cost)

                    heapq.heappush(priority_queue, neighbor_node)
                    visited[(path_cost, neighbor, neighbor_energy_cost)] = neighbor_node

        print("search failed for routing_request ", routing_request.routing_request_id)

    def search_with_added_obstacles(self, warehouse, routing_request, added_obstacles):
        """
        Treats coordinates in added_obstacles as static obstacles
        """
        source = self.source
        destination = self.destination
        destination_id = routing_request.destination.destination_id

        priority_queue = []
        heapq.heappush(priority_queue, source)
        visited = {source.vertex: source}

        while priority_queue:
            current_node = heapq.heappop(priority_queue)

            if current_node == destination:
                self.destination = current_node
                route = self.reconstruct_route()

                return route

            path_cost = current_node.g_value + 1
            if not EnvironmentUtils.is_energy_cost_valid(warehouse, path_cost):
                return None

            valid_neighbors = current_node.get_valid_neighbors_from_added_obstacles(added_obstacles)

            for neighbor in valid_neighbors:
                if neighbor in visited:
                    check_visited_neighbor(visited, neighbor, path_cost, current_node, priority_queue)
                else:
                    neighbor_node = AStar.Node(neighbor, neighbor.destination_distance[destination_id],
                                               path_cost, current_node, neighbor == source.vertex)
                    heapq.heappush(priority_queue, neighbor_node)
                    visited[neighbor] = neighbor_node

        return None

    def classic_astar(self, warehouse):
        """
        In this method the source and the destination may be any two warehouse nodes.
        Cartesian distance is used as the heuristic.
        """
        source = self.source
        destination = self.destination

        priority_queue = []
        heapq.heappush(priority_queue, source)
        visited = {source.vertex: source}

        while priority_queue:
            current_node = heapq.heappop(priority_queue)

            if current_node == destination:
                self.destination = current_node
                route = self.reconstruct_route()

                return route

            path_cost = current_node.g_value + 1
            if not EnvironmentUtils.is_energy_cost_valid(warehouse, path_cost):
                return None

            current_vertex = current_node.vertex

            neighbors = list(current_vertex.neighbors)
            if RANDOMIZE_NEIGHBORS:
                random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor in visited:
                    check_visited_neighbor(visited, neighbor, path_cost, current_node, priority_queue)
                else:
                    neighbor_node = AStar.Node(neighbor, distance(neighbor.coordinates, destination.vertex.coordinates),
                                               path_cost, current_node)

                    heapq.heappush(priority_queue, neighbor_node)
                    visited[neighbor] = neighbor_node

        return None


def check_visited_neighbor(visited, neighbor, path_cost, current_node, priority_queue):
    neighbor_node = visited[neighbor]
    if path_cost < neighbor_node.g_value:
        neighbor_node.parent = current_node
        neighbor_node.g_value = path_cost
        neighbor_node.f_value = path_cost + neighbor_node.h_value

        if neighbor_node not in priority_queue:
            heapq.heappush(priority_queue, neighbor_node)
        else:
            heapq.heapify(priority_queue)


def generate_astar_plan_for_first_routing_request(warehouse, routing_requests):
    routing_request = routing_requests[0]
    routing_request_vertex = routing_request.source
    source_node = AStar.Node(routing_request_vertex, routing_request_vertex.destination_distance[routing_request.destination.destination_id], 0,
                             None, True)
    destination_node = AStar.Node(routing_request.destination, 0, maxsize, None, False)
    a_star_framework = AStar(source_node, destination_node)
    route = a_star_framework.classic_astar(warehouse)
    plan = [route]
    return plan
