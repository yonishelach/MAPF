import os
import random
import numpy as np
import pandas as pd
from typing import Dict

from SingleSourceDestinationConflictsByNumberOfAgentsExperiment import generate_path_for_given_arrival_time
from Warehouse import Warehouse
from DatabaseInterface import initialize_database_preliminary_files

WH_SIZE = (40, 40)
MAX_DEVIATION_FACTOR = 1.7
NUM_AGENTS = 30


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

    parent_dir = f'{csv_target_dir}/src_{routing_request[0]}_dst_{routing_request[1]}'
    os.mkdir(parent_dir)
    lengths = set([len(x) for x in paths])
    num_paths = []
    for l in lengths:
        paths_by_length = [x for x in paths if len(x) == l]
        features = []
        _ = [features.extend([f'y_{idx}', f'x_{idx}']) for idx in range(l)]
        unpacked = [[item for t in pbl for item in t] for pbl in paths_by_length]
        df = pd.DataFrame(unpacked, columns=features).drop_duplicates()
        num_paths.append(df.shape[0])
        df.to_csv(f'{parent_dir}/path_length_{l}.csv', index=False)
    print(f'number of paths for each length:\n{num_paths}')
    print(f'Created a total of {sum(num_paths)} paths')
    print('\n----------------------------------------------------\n\n')


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

    for source in wh.sources:
        for destination in wh.destinations:
            create_paths_file_by_request(
                warehouse=wh,
                source_id=source.source_id,
                destination_id=destination.destination_id,
                size=1000,
                csv_target_dir='database/paths_by_routing_request'
            )
