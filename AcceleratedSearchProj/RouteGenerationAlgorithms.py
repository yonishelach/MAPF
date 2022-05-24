import random
from math import ceil
from sys import maxsize

from AStar import AStar
from BFS import BFS
from EnvironmentUtils import get_random_points_throughout_warehouse, is_valid_route_length, is_energy_cost_valid
from RoutingRequest import RoutingRequest
from Utils import distance

ROUTE_GENERATION_ALGORITHMS_ABBR = ["ROR", "k-ROR", "IPWS", "k-IPWS", "MPR", "k-MPR", "MPR_WS"]

PROGRESSIVELY_OBSTACLE_RESTRICTED_PLANS_MAX_TRIES = 5
OBSTACLE_PATTERNS = ["cross", "square", "vertical_line", "horizontal_line", "dot"]


def generate_midpoints_restricted_plan(warehouse, source, destination, is_split_at_midpoint=False):
    midpoints = get_random_points_throughout_warehouse(warehouse)
    midpoints.add(source.coordinates)

    plan = []
    for i, midpoint_coordinates in enumerate(midpoints):
        midpoint_vertex = warehouse.vertices[midpoint_coordinates[0]][midpoint_coordinates[1]]
        source_node = AStar.Node(source, distance(source.coordinates, midpoint_coordinates), 0, None, True)
        midpoint_node = AStar.Node(midpoint_vertex, 0, maxsize, None, False)
        a_star_framework = AStar(source_node, midpoint_node)

        route_to_midpoint = a_star_framework.classic_astar(warehouse)
        if not route_to_midpoint:
            continue

        source_node = AStar.Node(midpoint_vertex, midpoint_vertex.destination_distance[destination.destination_id], 0,
                                 None, True)
        destination_node = AStar.Node(destination, 0, maxsize, None, False)
        a_star_framework = AStar(source_node, destination_node)

        route_from_midpoint = a_star_framework.classic_astar(warehouse)
        if not route_from_midpoint:
            continue

        complete_route = route_to_midpoint + route_from_midpoint

        if is_valid_route_length(warehouse, route_to_midpoint + route_from_midpoint):
            if is_split_at_midpoint:
                # obstacle_patterns = ["cross", "square", "vertical_line", "horizontal_line", "dot"]
                obstacle_patterns = ["dot"]
                max_obstacle_size = min(warehouse.static_obstacle_length, warehouse.static_obstacle_width)
                split_step_size = max(2 * max_obstacle_size, 2)

                routing_request = RoutingRequest(i, midpoint_vertex, destination)
                for routing_request_route in generate_random_obstacles_restricted_plan(warehouse, routing_request,
                                                                                       obstacle_patterns,
                                                                                       split_step_size,
                                                                                       len(route_to_midpoint)):
                    if is_valid_route_length(warehouse, route_to_midpoint + routing_request_route):
                        plan.append(route_to_midpoint + routing_request_route)

            else:
                plan.append(complete_route)

    return plan


def generate_midpoints_restricted_plan_for_first_routing_request(warehouse, routing_requests,
                                                                 is_split_at_midpoint=False):
    first_routing_request = routing_requests[0]
    source, destination = first_routing_request.source, first_routing_request.destination

    return generate_midpoints_restricted_plan(warehouse, source, destination, is_split_at_midpoint)


def generate_ideal_path_with_splits_plan(warehouse, source, destination):
    ideal_path = (BFS(source, destination).generate_plan())[0]

    plan = [ideal_path]
    obstacle_patterns = ["cross", "square", "vertical_line", "horizontal_line", "dot"]
    max_obstacle_size = min(warehouse.static_obstacle_length, warehouse.static_obstacle_width)
    split_step_and_size = max_obstacle_size
    routing_request_id = 1
    for i, coordinates in enumerate(ideal_path):
        split_on_every_step = False
        if split_on_every_step or i % split_step_and_size == 0:
            routing_request = RoutingRequest(routing_request_id, warehouse.vertices[coordinates[0]][coordinates[1]],
                                             destination)
            for routing_request_route in generate_random_obstacles_restricted_plan(warehouse, routing_request,
                                                                                   obstacle_patterns,
                                                                                   4 * split_step_and_size, i):
                if not routing_request_route:
                    continue

                first_elements = ideal_path[:i - 1] if i != 0 else []
                if is_valid_route_length(warehouse, first_elements + routing_request_route):
                    plan.append(first_elements + routing_request_route)
            routing_request_id += 1

    return plan


def generate_ideal_path_with_splits_plan_for_first_routing_request(warehouse, routing_requests):
    first_routing_request = routing_requests[0]
    source, destination = first_routing_request.source, first_routing_request.destination

    return generate_ideal_path_with_splits_plan(warehouse, source, destination)


def add_obstacle_at_midpoint(added_obstacles, last_added_obstacle_midpoint, added_obstacle_size, obstacle_pattern):
    midpoint_x, midpoint_y = last_added_obstacle_midpoint[0], last_added_obstacle_midpoint[1]
    if obstacle_pattern == "cross":
        for i in range(added_obstacle_size):
            if (midpoint_x + ceil(added_obstacle_size / 2) - i, midpoint_y) not in added_obstacles:
                added_obstacles.add((midpoint_x + ceil(added_obstacle_size / 2) - i, midpoint_y))

            if (midpoint_x, midpoint_y + ceil(added_obstacle_size / 2) - i) not in added_obstacles:
                added_obstacles.add((midpoint_x, midpoint_y + ceil(added_obstacle_size / 2) - i))

    elif obstacle_pattern == "vertical_line":
        for i in range(2 * added_obstacle_size):
            if (midpoint_x + added_obstacle_size - i, midpoint_y) not in added_obstacles:
                added_obstacles.add((midpoint_x + added_obstacle_size - i, midpoint_y))

    elif obstacle_pattern == "horizontal_line":
        for i in range(2 * added_obstacle_size):
            if (midpoint_x, midpoint_y + added_obstacle_size - i) not in added_obstacles:
                added_obstacles.add((midpoint_x, midpoint_y + added_obstacle_size - i))

    else:  # square and dot pattern
        if obstacle_pattern == "dot":
            added_obstacle_size = 1

        for i in range(added_obstacle_size):
            if (midpoint_x + i, midpoint_y) not in added_obstacles:
                added_obstacles.add((midpoint_x + i, midpoint_y))

            if (midpoint_x, midpoint_y + i) not in added_obstacles:
                added_obstacles.add((midpoint_x, midpoint_y + i))

            if (midpoint_x + i, midpoint_y + added_obstacle_size - 1) not in added_obstacles:
                added_obstacles.add((midpoint_x + i, midpoint_y + added_obstacle_size - 1))

            if (midpoint_x + added_obstacle_size - 1, midpoint_y + i) not in added_obstacles:
                added_obstacles.add((midpoint_x + added_obstacle_size - 1, midpoint_y + i))


def generate_random_obstacles_restricted_plan(warehouse, routing_request, obstacle_patterns=None,
                                              max_routes=maxsize, initial_dist=0):
    # print("Generating random obstacles restricted plan, with obstacle patterns in", obstacle_patterns)
    # print("***")

    if obstacle_patterns is None:
        obstacle_patterns = OBSTACLE_PATTERNS
    plan = []
    added_obstacles = set()
    added_obstacles_backup = set()
    route_backup = []
    routing_request_source = routing_request.source
    max_added_obstacle_size = ceil(min(warehouse.static_obstacle_length, warehouse.static_obstacle_width))

    source_node = AStar.Node(routing_request_source,
                             routing_request_source.destination_distance[routing_request.destination.destination_id], 0,
                             None, True)
    destination_node = AStar.Node(routing_request.destination, 0, maxsize, None, False)
    a_star_framework = AStar(source_node, destination_node)

    tries = 0
    while len(plan) < max_routes and tries < PROGRESSIVELY_OBSTACLE_RESTRICTED_PLANS_MAX_TRIES:
        route = a_star_framework.search_with_added_obstacles(warehouse, routing_request, added_obstacles)

        if route and is_energy_cost_valid(warehouse, len(route) + initial_dist):
            tries = 0
            route_backup = route
            added_obstacles_backup = added_obstacles
            if is_valid_route_length(warehouse, route):
                plan.append(route)
            if len(plan) % 10 == 0:
                print("Still generating, generated", len(plan), "routes")

        else:
            tries += 1
            added_obstacles = added_obstacles_backup
            route = route_backup
        if not route:
            break

        obstacle_pattern = random.choice(obstacle_patterns)
        added_obstacle_size = random.randint(1, max_added_obstacle_size)
        min_idx = 4 * added_obstacle_size
        max_idx = len(route) - 1 - 4 * added_obstacle_size
        if max_idx - min_idx > 0:
            last_added_obstacle_midpoint = random.choice(route[4 * added_obstacle_size:-4 * added_obstacle_size])
        else:
            last_added_obstacle_midpoint = random.choice(route)

        if obstacle_pattern in {"square", "dot"}:
            last_added_obstacle_midpoint = (last_added_obstacle_midpoint[0] - ceil(added_obstacle_size / 2),
                                            last_added_obstacle_midpoint[1] - ceil(added_obstacle_size / 2))

        add_obstacle_at_midpoint(added_obstacles, last_added_obstacle_midpoint, added_obstacle_size,
                                 obstacle_pattern)
    return plan


def generate_random_obstacles_restricted_plan_for_first_routing_request(warehouse, routing_requests,
                                                                        obstacle_patterns=None):
    if obstacle_patterns is None:
        obstacle_patterns = OBSTACLE_PATTERNS
    first_routing_request = routing_requests[0]

    return generate_random_obstacles_restricted_plan(warehouse, first_routing_request, obstacle_patterns)


def generate_stupid_search_plan(warehouse, routing_requests, stupid_moves_left=0):
    stupid_moves_left = 2

    return
