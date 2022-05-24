import random
import numpy as np
import pandas as pd
from typing import Dict

from AcceleratedSearchProj import (
    generate_path_for_given_arrival_time,
    initialize_database_preliminary_files,
    Warehouse
)

WH_SIZE = (40, 40)
MAX_DEVIATION_FACTOR = 1.7
NUM_AGENTS = 30


def create_routing_requests(
        warehouse: Warehouse,
        maximal_deviation_factor: int = MAX_DEVIATION_FACTOR,
        agent_index: int = None
) -> Dict[str, int]:
    dict_keys = ['source', 'destination', 'arrival_time']
    if agent_index is not None:
        dict_keys = [f'{dk}_{agent_index}' for dk in dict_keys]
    # Generating random source and destination requests:
    sources = [s.source_id for s in warehouse.sources]
    destinations = [d.destination_id for d in warehouse.destinations]
    s = warehouse.sources[random.choice(sources)].coordinates[1]
    d = warehouse.destinations[random.choice(destinations)].coordinates[1]

    # Creating arrival time:
    minimal_arriving_time = warehouse.length + np.abs(s - d)
    maximal_arriving_time = minimal_arriving_time * maximal_deviation_factor
    arrival_time = np.random.randint(minimal_arriving_time, maximal_arriving_time)

    return {
        dict_keys[0]: s,
        dict_keys[1]: d,
        dict_keys[2]: arrival_time
    }


def create_training_dataset(
        warehouse: Warehouse,
        num_agents: int,
        size: int
):
    column_names = ['source', 'destination', 'arrival_time']
    columns = []
    for num_agent in range(num_agents):
        columns.extend([f'{cn}_{num_agent}' for cn in column_names])

    rr_df = pd.DataFrame(columns=columns)
    for _ in range(size):
        routing_requests = {}
        for agent in range(NUM_AGENTS):
            routing_requests.update(create_routing_requests(warehouse, agent_index=agent))
        rr_df = rr_df.append(routing_requests, ignore_index=True)

    rr_df.to_csv('./database/routing_requests.csv', index=False)


def create_path(
        warehouse: Warehouse,
        source_id: int,
        destination_id: int,
        maximal_deviation_factor: float = MAX_DEVIATION_FACTOR  # See if necessary
):
    ideal_path_length = warehouse.sources[source_id].destination_distance[destination_id]
    deviation_factor = random.uniform(1, maximal_deviation_factor)

    arrival_time = int(np.floor(ideal_path_length * deviation_factor))

    path = generate_path_for_given_arrival_time(
        warehouse=warehouse,
        source_id=source_id,
        destination_id=destination_id,
        arrival_time=arrival_time
    )

    return path


def create_paths_file_by_request(
        warehouse: Warehouse,
        source_id: int,
        destination_id: int,
        size: int,
        csv_target_dir: str
):
    routing_request = (warehouse.sources[source_id].coordinates[1], warehouse.destinations[destination_id].coordinates[1])
    paths = []
    print(f"Generating {size} paths for routing request: {routing_request}")
    for _ in range(size):
        path = create_path(
            warehouse=warehouse,
            source_id=source_id,
            destination_id=destination_id
        )
        paths.append(path)
    df = pd.DataFrame()
    df['path'] = paths

    df.to_csv(f'{csv_target_dir}/{routing_request[0]}_{routing_request[1]}.csv', index=False)


# Example for creating Database and route request dataset:
if __name__ == '__main__':
    length, width = WH_SIZE
    wh = Warehouse(
        warehouse_id=1000,
        length=length,
        width=width,
        number_of_sources=20,
        number_of_destinations=10,
        static_obstacle_length=round(0.1 * length),
        static_obstacle_width=round(0.1 * width),
        static_obstacle_layout=[],
        is_warehouse_searchable=True
    )
    initialize_database_preliminary_files(wh)

    for source in wh.sources:
        for destination in wh.destinations:
            create_paths_file_by_request(
                warehouse=wh,
                source_id=source.source_id,
                destination_id=destination.destination_id,
                size=1000,
                csv_target_dir='./database/paths_by_routing_request'
            )

    create_training_dataset(warehouse=wh, num_agents=NUM_AGENTS, size=1000)
