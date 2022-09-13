import pandas as pd
import random
import numpy as np
from typing import List
# def create_csv_file(
#         directory_path: str,
#         warehouse: Warehouse = warehouse_model,
#         num_agents: int = NUM_AGENTS,
#         size: int = 10000,
# ):
#     # Gathering all possible src and dst combinations:
#     rr_without_time = []
#     for source in warehouse.sources:
#         for destination in warehouse.destinations:
#             rr_without_time.append([source.coordinates[1], destination.coordinates[1]])
#
#     column_names = ['source', 'destination']
#     columns = []
#     for num_agent in range(num_agents):
#         columns.extend([f'{cn}_{num_agent}' for cn in column_names])
#     rr_df = pd.DataFrame(columns=columns)
#
#     for _ in range(size):
#         rr = sorted(random.choices(rr_without_time, k=num_agents))
#         rr_unpacked = [item for t in rr for item in t]
#         rr_df.loc[len(rr_df)] = rr_unpacked
#
#     rr_df = rr_df.drop_duplicates()
#     if not os.path.exists(directory_path):
#         os.mkdir(directory_path)
#     target_path = f'./{directory_path}/routing_requests_without_time.csv'
#     rr_df.to_csv(target_path, index=False)
#     print(f'Created routing requests without time successfully in {target_path}')


MAX_ITERATION = 200


def count_collisions(plan: List[np.ndarray]):
    # positions = np.zeros((*WAREHOUSE_SIZE, MAX_LEN), dtype=int)
    positions = {}
    # in the same place at the same time:
    num_vertex_conflict = 0
    # switching places (the same "edge" at the same time):
    num_swapping_conflicts = 0

    # Changing shape to
    # plan = plan.reshape((NUM_RR, -1, 2))

    for agent in range(len(plan)):

        path = plan[agent]
        prev_location = path[0]

        for time in range(len(path)):
            location = path[time]
            if all(location == path[0]) or all(location == path[-1]) or all(location == -1):  # agent is at source or at destination
                continue

            # add position to visit map:
            prev_pos_key = f'{prev_location[0]}_{location[1]}'
            pos_key = f'{location[0]}_{location[1]}'
            if pos_key not in positions:
                positions[pos_key] = {}
            if time not in positions[pos_key]:
                positions[pos_key][time] = []
            else:
                num_vertex_conflict += len(positions[pos_key][time])

            # find swapping conflicts:
            if all(prev_location != path[0]):
                if (time in positions[prev_pos_key]) and (time - 1 in positions[pos_key]):
                    for other_agent in positions[pos_key][time - 1]:
                        if other_agent in positions[prev_pos_key][time]:
                            num_swapping_conflicts += 1

            positions[pos_key][time].append(agent)
            prev_location = location
    return num_vertex_conflict + num_swapping_conflicts


def time_range(start, end):
    t_min = 10 + np.abs(start - end)
    t_max = int((t_min - 1) * 1.7)
    return t_min, t_max


if __name__ == '__main__':
    examples = []
    solutions = []
    while len(examples) < 500:
        points = [0, 2, 4, 6, 8]
        choices = []
        for _ in range(8):
            choice = [random.choice(points), random.choice(points)]
            choice.append(random.choice(time_range(*choice)))
            while choice in choices:
                choice = [random.choice(points), random.choice(points)]
                choice.append(random.choice(time_range(*choice)))
            # print('*', end='')
            choices.append(choice)
        if choices in examples:
            continue
        # print()
        data_dir = './data_new'
        requests = ["src_{}_dst_{}/path_length_{}.csv".format(*choice) for choice in choices]
        routes = {}
        for i, req in enumerate(requests):
            routes[i] = pd.read_csv(f'{data_dir}/routes_by_request/{req}')
        plan = []
        for x in range(MAX_ITERATION):
            for route in routes.values():
                idx = random.choice(range(len(route)))
                chosen_route = route.loc[idx].values.reshape(-1, 2)
                plan.append(chosen_route)
            num_collisions = count_collisions(plan)
            if num_collisions:
                plan = []
            else:
                examples.append(choices)
                solutions.append(plan)
                if len(examples) % 10 == 0:
                    print()
                    print(f'Found {len(examples)} examples')
                else:
                    print('.', end='')
                break

    cols = ['x_src', 'y_src', 'x_dst', 'y_dst', 'time']
    columns = []
    for i in range(8):
        columns.extend([f'{col}_{i}' for col in cols])
    examples_df = pd.DataFrame(columns=columns)
    solutions_df = pd.DataFrame(columns=['solution'])
    for example in examples:
        for i in range(len(example)):
            ex = example[i]
            ex.insert(1, 0)
            ex.insert(3, 0)
            example[i] = ex
    for example, solution in zip(examples, solutions):
        unpacked = [item for sublist in example for item in sublist]
        examples_df.loc[len(examples_df)] = unpacked
        solutions_df.loc[len(solutions_df)] = str([sol.tolist() for sol in solution])
    examples_df.to_csv('examples.csv', index=False)
    solutions_df.to_csv('solutions.csv', index=False)
