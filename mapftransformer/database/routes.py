import random
import os
import numpy as np
import pandas as pd
from ..warehouse import Warehouse, warehouse_model
from .utils import generate_path_for_given_arrival_time

MAX_DEVIATION_FACTOR = 1.7


class Route:
    """
    This class posses the ability to create a database of
    """

    @staticmethod
    def _create_route(
            warehouse: Warehouse,
            source_id: int,
            destination_id: int,
            maximal_deviation_factor: float = MAX_DEVIATION_FACTOR
    ):
        """
        Generating a route in the warehouse from source to destination based on the midpoint algorithm.

        :param warehouse:                   The relevant warehouse. This warehouse must contain the destination id and the source id.
        :param source_id:                   The id number of the source where the route will start from.
        :param destination_id:              The id number of the destination where the route will end.
        :param maximal_deviation_factor:    The deviation factor, a float number, which the higher it gets, the route can be more twisted.
                                            Defaulted to 1.7.
        :return:    A route, which is a list of coordinates in the warehouse.
        """
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

    def _create_csv_files_by_request(
            self,
            warehouse: Warehouse,
            source_id: int,
            destination_id: int,
            size: int,
            csv_target_dir: str
    ):
        """
        Create a csv files by a routing request. The csv files are divided by length. The name of each csv file 'path_length_<length>.csv'.
        Each csv file will have the following headers: y_0, x_0, ..., y_{i}, x_{i}, where i = length - 1.
        All the csv files will be stored in `<csv_target_dir>`/src_<x_0>_dst_<x_{i}> folder.

        :param warehouse:       The relevant warehouse.
        :param source_id:       The id number of the source where the routes will start from.
        :param destination_id:  The id number of the destination where the routes will end.
        :param size:            The maximum number of routes to generate for the given routing request.
        :param csv_target_dir:  The directory path to save the routes in.

        :return:    None.
        """

        routing_request = (warehouse.sources[source_id].coordinates[1], warehouse.destinations[destination_id].coordinates[1])
        routes = []
        print(f"Generating at most {size} paths for routing request: {routing_request}")
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
        """
        Creating the dataset for routes by requests for the given warehouse. The structure of the dataset is as follows:
        Each folder will be named 'src_i_dst_j', which will contain all the CSV files that contain
        routes that begin at source coordinate "i" and end in destination coordinate "j".
        Each folder will possess several CSV files that are divided by the routes' length.

        For example, a route with length 18 from source 6 to destination 2 will be stored in:

            "<directory_path>/src_6_dst_2/path_length_18.csv"

        :param directory_path:  The target path to save all of these files.
        :param warehouse:       The relevant warehouse.
        :param size:            The maximum number of routes to generate for the given routing request.

        :return:    None.
        """
        warehouse.initialize_database_preliminary_files()
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)

        for source in warehouse.sources:
            for destination in warehouse.destinations:
                self._create_csv_files_by_request(
                    warehouse=warehouse,
                    source_id=source.source_id,
                    destination_id=destination.destination_id,
                    size=size,
                    csv_target_dir=f'{directory_path}'
                )
