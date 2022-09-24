import random
import os
import numpy as np
import pandas as pd
from ..warehouse import Warehouse, warehouse_model
from .utils import generate_path_for_given_arrival_time

MAX_DEVIATION_FACTOR = 1.7


class Route:

    @staticmethod
    def _create_route(
            warehouse: Warehouse,
            source_id: int,
            destination_id: int,
            maximal_deviation_factor: float = MAX_DEVIATION_FACTOR
    ):
        ideal_path_length = warehouse.sources[source_id].destination_distance[destination_id]
        deviation_factor = random.uniform(1, maximal_deviation_factor)

        arrival_time = int(np.floor(ideal_path_length * deviation_factor))

        route = generate_path_for_given_arrival_time(
            warehouse=warehouse,
            source_id=source_id,
            destination_id=destination_id,
            arrival_time=arrival_time
        )

        return route

    def _create_single_csv_file_by_request(
            self,
            warehouse: Warehouse,
            source_id: int,
            destination_id: int,
            size: int,
            csv_target_dir: str
    ):
        routing_request = (warehouse.sources[source_id].coordinates[1], warehouse.destinations[destination_id].coordinates[1])
        routes = []
        print(f"Generating {size} paths for routing request: {routing_request}")
        for _ in range(size):
            route = self._create_route(
                warehouse=warehouse,
                source_id=source_id,
                destination_id=destination_id
            )
            routes.append(route)

        parent_dir = f'{csv_target_dir}/src_{routing_request[0]}_dst_{routing_request[1]}'
        os.mkdir(parent_dir)
        lengths = set([len(x) for x in routes])
        lengths = sorted(list(lengths))
        num_paths = []
        for length in lengths:
            routes_by_length = [x for x in routes if len(x) == length]
            features = []
            _ = [features.extend([f'y_{idx}', f'x_{idx}']) for idx in range(length)]
            unpacked = [[item for t in rbl for item in t] for rbl in routes_by_length]

            # Need to add here routes with previous lower length with waiting in the start:
            if length - 1 in lengths:
                prev_df = pd.read_csv(f'{parent_dir}/path_length_{length - 1}.csv')
                values = prev_df.values.tolist()
                wait_value = values[0][:2]
                values_with_waiting = [wait_value + v for v in values]
                unpacked += values_with_waiting
            df = pd.DataFrame(unpacked, columns=features).drop_duplicates()
            num_paths.append(df.shape[0])
            df.to_csv(f'{parent_dir}/path_length_{length}.csv', index=False)
        print(f'number of paths for each length:\n{num_paths}')
        print(f'Created a total of {sum(num_paths)} paths')
        print('\n----------------------------------------------------\n\n')

    def create_csv_files(
            self,
            directory_path: str,
            warehouse: Warehouse = warehouse_model,
            size: int = 1000,
    ):
        warehouse.initialize_database_preliminary_files()
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)

        for source in warehouse.sources:
            for destination in warehouse.destinations:
                self._create_single_csv_file_by_request(
                    warehouse=warehouse,
                    source_id=source.source_id,
                    destination_id=destination.destination_id,
                    size=size,
                    csv_target_dir=f'{directory_path}'
                )
