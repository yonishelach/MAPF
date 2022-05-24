import random

from RoutingRequest import RoutingRequest
from Utils import update_plan

RANDOMIZE_NEIGHBORS = True


class BFS:
    """
    Best First Search
    """
    def __init__(self, source, destination, vertex=None):
        self.source = source
        self.destination = destination
        self.vertex = self.source if not vertex else vertex

    def get_destination_distance(self):
        return self.vertex.destination_distance[self.destination.destination_id]

    def get_ideal_neighbor(self):
        ideal_neighbor = None
        min_destination_distance = self.get_destination_distance() + 1

        neighbors = list(self.vertex.neighbors)
        if RANDOMIZE_NEIGHBORS:
            random.shuffle(neighbors)

        for neighbor in neighbors:
            neighbor_destination_distance = neighbor.get_destination_distance(self.destination.destination_id)
            if neighbor_destination_distance < min_destination_distance:
                return neighbor

    def search(self):
        route = [self.vertex.coordinates]

        while self.vertex != self.destination:
            self.vertex = self.get_ideal_neighbor()
            route.append(self.vertex.coordinates)

        return route

    def generate_plan(self):
        plan = [[]]

        route = self.search()
        update_plan(plan, 0, route)
        return plan


def generate_bfs_plan(warehouse, routing_request):
    bfs = BFS(routing_request.source, routing_request.destination)

    return bfs.generate_plan()


def generate_bfs_plan_for_first_routing_request(warehouse, routing_requests):
    return generate_bfs_plan(warehouse, routing_requests[0])
