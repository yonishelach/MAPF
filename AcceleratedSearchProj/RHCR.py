from CBS.CBS import CBS


class RHCR:
    """
    window - the plan is free of conflicts until window, in other words the CBS will take into account all conflicts
             in window.
    time_to_plan - time to run CBS again, this is the actual window size that added to the solution each iteration.
    RHCR run CBS in iterations. Each iteration CBS considers only the conflicts inside window.
    Then, RHCR takes only part of the solution for each route - until time_to_plan.
    """
    def __init__(self, window, time_to_plan):
        self.window = window
        self.time_to_plan = time_to_plan

    def generate_rhcr_plan(self, warehouse, routing_requests):
        if self.time_to_plan < 1 or self.window < self.time_to_plan:
            print("non valid window/time_to_plan values.")
            print("time to plan has to be at least 1.")
            print("window has to be greater than or equal to time to plan.")
            exit(1)
        plan = [[routing_requests[i].source.coordinates] for i in range(len(routing_requests))]
        routes_indexes_to_plan = [i for i in range(len(routing_requests))]
        counter = 1
        while routes_indexes_to_plan:
            remaining_routes = [routing_requests[route_index] for route_index in routes_indexes_to_plan]
            cbs = CBS()
            new_plan = cbs.solve(warehouse, remaining_routes, self.window)
            indexes_to_remove = []
            for i, route_index in enumerate(routes_indexes_to_plan):
                for j in range(1, min(self.time_to_plan, len(new_plan[i]))):
                    plan[route_index].append(new_plan[i][j])
                if plan[route_index][-1] == routing_requests[route_index].destination.coordinates:
                    indexes_to_remove.append(route_index)
                else:
                    new_source_coordinates = plan[route_index][-1]
                    routing_requests[route_index].source = warehouse.vertices[new_source_coordinates[0]][new_source_coordinates[1]]
            for index in indexes_to_remove:
                routes_indexes_to_plan.remove(index)
            for route in plan:
                print(route)
            is_deadlock = deadlock_detector(plan, warehouse, routing_requests, counter)
            if is_deadlock:
                self.time_to_plan = self.time_to_plan+1
                self.window = self.window+1
                print(f"The time_to_plan increased by 1 and now is: {self.time_to_plan}")
                print(f"The window increased by 1 and now is: {self.window}")
            counter += 1
        return plan


def deadlock_detector(plan, warehouse, routing_requests, counter):
    """
    Deals only with deadlocks caused by robots that wait in sources.
    If wait outside the warehouse' sources is allowed, need to implement deadlock detector to deal with that.
    """
    number_of_agents_not_at_sources = 0
    for route in plan:
        coordinates = route[-1]
        vertex_node = warehouse.vertices[coordinates[0]][coordinates[1]]
        if vertex_node not in warehouse.sources:
            number_of_agents_not_at_sources += 1
    sources_num = len(warehouse.sources)
    if number_of_agents_not_at_sources == len(routing_requests):
        return False
    if number_of_agents_not_at_sources < sources_num*counter:
        return True
    return False
