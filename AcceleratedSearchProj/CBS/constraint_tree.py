from typing import Dict
import numpy as np
from ..RoutingRequest import RoutingRequest
from constraints import Constraints


def remove_duplicate_destinations_from_sol(route):
    if any(route[-1] != route[-2]):
        return route
    for i, (item1, item2) in enumerate(zip(route[::-1], route[:-1][::-1])):
        if any(item1 != item2):
            return route[:-i]


class CTNode:

    def __init__(self, constraints: Constraints, solution: Dict[RoutingRequest, np.ndarray]):
        self.constraints = constraints
        self.solution = solution
        self.cost = self.sic(solution)

    # Sum-of-Individual-Costs heuristics
    @staticmethod
    def sic(solution):
        return sum(len(remove_duplicate_destinations_from_sol(sol[1])) for sol in solution.items())

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return str(self.constraints.agent_constraints)
