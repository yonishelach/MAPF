import random
from time import time
#from CBS.CBS import CBS
import NoDeviationFactorDatabase
# from NoDeviationFactorDatabase import sample_routing_request_plan_from_database
from CBS.CBS import CBS
from AStar import generate_astar_plan_for_first_routing_request
from EnvironmentUtils import generate_rand_routing_requests, generate_rand_routing_request
from BFS import generate_bfs_plan_for_first_routing_request
from RND import generate_rnd_plan
from LNS_PP import generate_lns_rnd_plan
from RouteGenerationAlgorithms import generate_ideal_path_with_splits_plan_for_first_routing_request, \
    generate_midpoints_restricted_plan_for_first_routing_request, \
    generate_random_obstacles_restricted_plan_for_first_routing_request, ROUTE_GENERATION_ALGORITHMS_ABBR, \
    generate_stupid_search_plan
from RoutingRequest import RoutingRequest
from RHCR import RHCR

# NEIGHBORHOOD_SIZE is const for LNS_RND algorithm
NEIGHBORHOOD_SIZE = 15
WAVES_PER_WAREHOUSE = [2, 2, 5, 1]
K_SAMPLE_SIZE = 5


def generate_plan(warehouse, routing_requests, plan_generation_algorithm, optional_argument=None):
    """
    is_split_at_midpoint used for midpoints_restricted only
    """
    t0 = time()
    if optional_argument:
        plan = plan_generation_algorithm(warehouse, routing_requests, optional_argument)
    else:
        plan = plan_generation_algorithm(warehouse, routing_requests)
    t1 = time()
    return plan, t0, t1


def generate_stupid_search_example(warehouse, routing_requests, stupid_moves_left=0):
    return generate_plan(warehouse, routing_requests, generate_stupid_search_plan, stupid_moves_left)


def generate_astar_example(warehouse, routing_requests):
    return generate_plan(warehouse, routing_requests, generate_astar_plan_for_first_routing_request)


def generate_sample_database_avoiding_conflicts_example(warehouse, routing_requests):
    return generate_plan(warehouse, routing_requests, NoDeviationFactorDatabase.sample_routing_database_avoiding_conflicts)


def generate_sample_database_example(warehouse, routing_requests):
    return generate_plan(warehouse, routing_requests, NoDeviationFactorDatabase.sample_routing_request_plan_from_database)


def generate_midpoints_restricted_example(warehouse, routing_requests, is_split_at_midpoint=False):
    return generate_plan(warehouse, routing_requests, generate_midpoints_restricted_plan_for_first_routing_request,
                         is_split_at_midpoint)


def generate_midpoints_restricted_with_splits_example(warehouse, routing_requests):
    return generate_midpoints_restricted_example(warehouse, routing_requests, True)


def generate_ideal_path_with_splits_example(warehouse, routing_requests):
    return generate_plan(warehouse, routing_requests, generate_ideal_path_with_splits_plan_for_first_routing_request)


def generate_random_obstacles_restricted_example(warehouse, routing_requests):
    return generate_plan(warehouse, routing_requests,
                         generate_random_obstacles_restricted_plan_for_first_routing_request)


def generate_lns_rnd_example(warehouse, routing_requests, neighborhood_size):
    return generate_plan(warehouse, routing_requests, generate_lns_rnd_plan, neighborhood_size)


def generate_rnd_example(warehouse, routing_requests):
    return generate_plan(warehouse, routing_requests, generate_rnd_plan)


def generate_bfs_example(warehouse, routing_requests):
    return generate_plan(warehouse, routing_requests, generate_bfs_plan_for_first_routing_request)


def generate_cbs_example(warehouse, routing_requests):
    cbs = CBS()
    return generate_plan(warehouse, routing_requests, cbs.solve)


def generate_rhcr_example(warehouse, routing_requests, window, time_to_plan):
    rhcr = RHCR(window, time_to_plan)
    return generate_plan(warehouse, routing_requests, rhcr.generate_rhcr_plan)


def generate_example(warehouse, algorithm_name, routing_requests_in_tuples_format=None, window=0, time_to_plan=0, neighborhood_size=NEIGHBORHOOD_SIZE):
    """
    if no value for routing_requests_in_tuples_format is supplied, a random value for routing_requests_in_tuples_format
    is generated
    """

    if routing_requests_in_tuples_format:
        if algorithm_name in ROUTE_GENERATION_ALGORITHMS_ABBR:
            routing_requests_in_tuples_format = [routing_requests_in_tuples_format[0]]

        routing_requests = [RoutingRequest(i, warehouse.sources[routing_request[0]],
                                           warehouse.destinations[routing_request[1]]) for i, routing_request
                            in enumerate(routing_requests_in_tuples_format)]
    else:
        if algorithm_name in ROUTE_GENERATION_ALGORITHMS_ABBR:
            routing_requests = generate_rand_routing_request(warehouse)
        else:
            routing_requests = generate_rand_routing_requests(warehouse, WAVES_PER_WAREHOUSE)
        routing_requests_in_tuples_format = [(routing_request.get_source_id(), routing_request.get_destination_id())
                                             for routing_request in routing_requests]

    # print(routing_requests_in_tuples_format)
    if algorithm_name == "BFS":
        plan, t0, t1 = generate_bfs_example(warehouse, routing_requests)

    elif algorithm_name == "RND":
        plan, t0, t1 = generate_rnd_example(warehouse, routing_requests)

    elif algorithm_name == "LNS_RND":
        plan, t0, t1 = generate_lns_rnd_example(warehouse, routing_requests, neighborhood_size)

    elif algorithm_name == "CBS":
        plan, t0, t1 = generate_cbs_example(warehouse, routing_requests)

    elif algorithm_name == "RHCR":
        plan, t0, t1 = generate_rhcr_example(warehouse, routing_requests, window, time_to_plan)

    elif algorithm_name == "ROR":
        plan, t0, t1 = generate_random_obstacles_restricted_example(warehouse, routing_requests)

    elif algorithm_name == "k-ROR":
        plan, t0, t1 = generate_random_obstacles_restricted_example(warehouse, routing_requests)
        plan = random.sample(plan, K_SAMPLE_SIZE)

    elif algorithm_name == "IPWS":
        plan, t0, t1 = generate_ideal_path_with_splits_example(warehouse, routing_requests)

    elif algorithm_name == "k-IPWS":
        plan, t0, t1 = generate_ideal_path_with_splits_example(warehouse, routing_requests)
        plan = random.sample(plan, K_SAMPLE_SIZE)

    elif algorithm_name == "MPR":
        plan, t0, t1 = generate_midpoints_restricted_example(warehouse, routing_requests)

    elif algorithm_name == "k-MPR":
        plan, t0, t1 = generate_midpoints_restricted_example(warehouse, routing_requests)
        plan = random.sample(plan, K_SAMPLE_SIZE)

    elif algorithm_name == "MPR_WS":
        plan, t0, t1 = generate_midpoints_restricted_with_splits_example(warehouse, routing_requests)

    elif algorithm_name == "k-MPR_WS":
        plan, t0, t1 = generate_midpoints_restricted_with_splits_example(warehouse, routing_requests)
        plan = random.sample(plan, K_SAMPLE_SIZE)

    elif algorithm_name == "sample_database":
        plan, t0, t1 = generate_sample_database_example(warehouse, routing_requests_in_tuples_format)

    elif algorithm_name == "sample_database_avoiding_conflicts":
        plan, t0, t1 = generate_sample_database_avoiding_conflicts_example(warehouse, routing_requests_in_tuples_format)

    elif algorithm_name == 'AStar':
        plan, t0, t1 = generate_astar_example(warehouse, routing_requests)

    else:
        print("Unsupported algorithm_name.\n", "Currently supports:", "BFS, RND, LNS_RND")
        return [[]], -1

    running_time = round(t1 - t0, 4)
    return plan, running_time, routing_requests_in_tuples_format
