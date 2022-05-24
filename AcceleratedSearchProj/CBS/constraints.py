from typing import Dict, Tuple, Set
from ..RoutingRequest import RoutingRequest


class Constraints:

    def __init__(self):
        #                                   time,         obstacles
        self.agent_constraints: Dict[RoutingRequest: Dict[int, Set[Tuple[int, int]]]] = dict()

    '''
    Deepcopy self with additional constraints
    '''

    def fork(self, agent: RoutingRequest, obstacle: Tuple[int, int], start: int) -> 'Constraints':
        agent_constraints_copy = self.agent_constraints.copy()
        agent_constraints_copy.setdefault(agent, dict()).setdefault(start, set()).add(obstacle)
        new_constraints = Constraints()
        new_constraints.agent_constraints = agent_constraints_copy
        return new_constraints

    def setdefault(self, key, default):
        return self.agent_constraints.setdefault(key, default)

    def __getitem__(self, agent):
        return self.agent_constraints[agent]

    def __iter__(self):
        for key in self.agent_constraints:
            yield key

    def __str__(self):
        return str(self.agent_constraints)
