import os
import random
import pandas as pd

from ..warehouse import Warehouse, NUM_AGENTS, warehouse_model


class RoutingRequest:

    @staticmethod
    def create_csv_file(
            directory_path: str,
            warehouse: Warehouse = warehouse_model,
            num_agents: int = NUM_AGENTS,
            size: int = 10000,
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

        for _ in range(size):
            rr = sorted(random.choices(rr_without_time, k=num_agents))
            rr_unpacked = [item for t in rr for item in t]
            rr_df.loc[len(rr_df)] = rr_unpacked

        rr_df = rr_df.drop_duplicates()
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        target_path = f'./{directory_path}/routing_requests_without_time.csv'
        rr_df.to_csv(target_path, index=False)
        print(f'Created routing requests without time successfully in {target_path}')
