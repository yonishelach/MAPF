from copy import deepcopy
from itertools import combinations
from typing import List, Tuple, Dict
import numpy as np
from constraint_tree import CTNode
from constraints import Constraints
from ..RoutingRequest import RoutingRequest
from ..EnvironmentUtils import find_route_using_Astar
from heapq import heappush, heappop
from tqdm import tqdm

from ..Warehouse import Warehouse


class CBS:
    def __init__(self):
        self.agents = None
        self.warehouse = None
        self.robot_radius = 1

    '''
        You can use your own assignment function, the default algorithm greedily assigns
        the closest goal to each start.
    '''

    def solve(self, warehouse: Warehouse, agents: List[RoutingRequest], window=float('inf')):
        self.warehouse = warehouse
        self.agents: List[RoutingRequest] = agents
        plan: List[List[Tuple[int, int]]] = [[] for _ in range(len(self.agents))]

        agent_to_route_dict = dict(
            (agent, np.asarray(find_route_using_Astar(self.warehouse, plan, agent))) for agent in self.agents)

        constraints = Constraints()

        # Build conflict tree
        open = []
        if all(len(path) != 0 for path in agent_to_route_dict.values()):
            # Make root node
            node = CTNode(constraints, agent_to_route_dict)
            # Min heap for quick extraction
            open.append(node)

        with tqdm(total=0, desc="Running CBS...") as progress_bar:
            while open:
                progress_bar.update(1)
                results = []
                self.search_node(heappop(open), results, plan, window)
                for result in results:
                    if len(result) == 1:
                        print("CBS Finished")
                        return self.format_result(result[0])
                    if result[0]:
                        heappush(open, result[0])
                    if result[1]:
                        heappush(open, result[1])
        print("CBS Failed")
        return plan

    def format_result(self, result):
        plan = []
        for route in result:
            plan.append([tuple(coordinate) for coordinate in route])
        return plan

    def search_node(self, best: CTNode, results, plan, window: int):
        agent_i, agent_j, time_of_conflict = self.validate_paths(self.agents, best, window)

        # If there is no conflict, validate_paths returns (None, None, -1)
        if agent_i is None:
            results.append((self.reformat(self.agents, best.solution),))
            return
        # Calculate new constraints
        agent_i_constraint = self.calculate_constraints(best, agent_i, agent_j, time_of_conflict)
        agent_j_constraint = self.calculate_constraints(best, agent_j, agent_i, time_of_conflict)

        # Calculate new paths
        agent_i_path = find_route_using_Astar(self.warehouse, plan, agent_i,
                                              constraints=agent_i_constraint.agent_constraints[agent_i])
        agent_j_path = find_route_using_Astar(self.warehouse, plan, agent_j,
                                              constraints=agent_j_constraint.agent_constraints[agent_j])
        # print(agent_i_path)
        # print(best.solution[agent_j])
        # print(agent_j_path)
        # print(best.solution[agent_i])
        # Replace old paths with new ones in solution
        solution_i = best.solution
        solution_j = self.deep_copy_solution(best.solution)
        solution_i[agent_i] = np.asarray(agent_i_path)
        solution_j[agent_j] = np.asarray(agent_j_path)

        node_i = None
        if all(len(path) != 0 for path in solution_i.values()):
            node_i = CTNode(agent_i_constraint, solution_i)

        node_j = None
        if all(len(path) != 0 for path in solution_j.values()):
            node_j = CTNode(agent_j_constraint, solution_j)

        results.append((node_i, node_j))

    def deep_copy_solution(self, solution: Dict[RoutingRequest, np.ndarray]):
        solution_copy = {}
        for key, value in solution.items():
            solution_copy[key] = deepcopy(value)
        return solution_copy

    '''
    Pair of agent, point of conflict
    '''

    def validate_paths(self, agents, node: CTNode, window: int):
        # Check collision pair-wise
        for agent_i, agent_j in combinations(agents, 2):
            time_of_conflict = self.safe_distance(node.solution, agent_i, agent_j, window)
            # time_of_conflict=-1 if there is no conflict
            if time_of_conflict == -1:
                continue
            return agent_i, agent_j, time_of_conflict
        return None, None, -1

    def safe_distance(self, solution: Dict[RoutingRequest, np.ndarray], agent_i: RoutingRequest,
                      agent_j: RoutingRequest, window: int) -> int:
        i_prev: Tuple[int, int]
        j_prev: Tuple[int, int]
        for idx, (point_i, point_j) in enumerate(zip(solution[agent_i], solution[agent_j])):
            if idx >= window:
                break
            if (all(point_i == agent_i.source.coordinates) and all(point_j == agent_j.source.coordinates)) \
                    or (all(point_i == agent_i.destination.coordinates) or all(point_j == agent_j.destination.coordinates))\
                :
                i_prev = point_i
                j_prev = point_j
                continue
            if (idx > 0 and (all(point_i == j_prev) and all(point_j == i_prev))):
                return idx
            if any(point_i != point_j) \
                    :
                i_prev = point_i
                j_prev = point_j
                continue
            return idx
        return -1

    @staticmethod
    def dist(point1: np.ndarray, point2: np.ndarray) -> int:
        return int(np.linalg.norm(point1 - point2, 2))  # L2 norm

    def calculate_constraints(self, node: CTNode,
                              constrained_agent: RoutingRequest,
                              unchanged_agent: RoutingRequest,
                              time_of_conflict: int) -> Constraints:
        constrained_path = node.solution[constrained_agent]
        unchanged_path = node.solution[unchanged_agent]

        pivot = constrained_path[time_of_conflict]
        return node.constraints.fork(constrained_agent, tuple(pivot.tolist()), time_of_conflict)

    '''
    Reformat the solution to a numpy array
    '''

    @staticmethod
    def reformat(agents: List[RoutingRequest], solution: Dict[RoutingRequest, np.ndarray]):
        solution = CBS.pad(solution)
        reformatted_solution = []
        for agent in agents:
            reformatted_solution.append(solution[agent])
        return np.array(reformatted_solution)

    '''
    Pad paths to equal length, inefficient but well..
    '''

    @staticmethod
    def pad(solution: Dict[RoutingRequest, np.ndarray]):
        max_ = max(len(path) for path in solution.values())
        for agent, path in solution.items():
            if len(path) == max_:
                continue
            padded = np.concatenate([path, np.array(list([path[-1]]) * (max_ - len(path)))])
            solution[agent] = padded
        return solution
