import os
import random
import numpy as np
import pandas as pd
from typing import Dict, List

from Warehouse import Warehouse
from DatabaseInterface import initialize_database_preliminary_files

WH_SIZE = (40, 40)
MAX_DEVIATION_FACTOR = 1.7
NUM_AGENTS = 30
NUM_RR = 10000
COLUMNS = ['source', 'destination', 'arrival_time']


def create_routing_requests(
        warehouse: Warehouse,
        maximal_deviation_factor: int = MAX_DEVIATION_FACTOR,
        agent_index: int = None,
) -> Dict[str, int]:
    dict_keys = COLUMNS
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


def create_validation_dataset(
        warehouse: Warehouse,
        num_agents: int,
        size: int,
):
    column_names = COLUMNS
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


def create_training_dataset(
        warehouse: Warehouse,
        num_agents: int = NUM_AGENTS,
        num_rr: int = NUM_RR,

):
    # Gathering all possible src and dst combinations:
    rr_without_time = []
    for source in warehouse.sources:
        for destination in warehouse.destinations:
            rr_without_time.append([source.coordinates[1], destination.coordinates[1]])

    column_names = ['source', 'destination']
    columns = []
    for num_agent in range(num_agents):
        columns.extend([f'{cn}_{num_agent}' for cn in column_names])
    rr_df = pd.DataFrame(columns=columns)

    for _ in range(num_rr):
        rr = sorted(random.choices(rr_without_time, k=num_agents))
        rr_unpacked = [item for t in rr for item in t]
        rr_df.loc[len(rr_df)] = rr_unpacked

    rr_df = rr_df.drop_duplicates()
    rr_df.to_csv('./database/routing_requests_without_time.csv', index=False)


# Example for creating Database and routing request dataset:
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

    # # With arriving time:
    # create_validation_dataset(warehouse=wh, num_agents=NUM_AGENTS, size=1000)

    create_training_dataset(warehouse=wh)


