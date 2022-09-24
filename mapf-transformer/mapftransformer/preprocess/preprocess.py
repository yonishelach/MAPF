from typing import Tuple, Optional, List
import os
import numpy as np
import pandas as pd
import json
import textwrap
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from mapftransformer.warehouse import Warehouse, warehouse_model
import itertools

class Moves:
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class RouteEncoder:
    """
    The encoding logic:
    - The number divided to 3 digits blocks.
    - Each part (contains at most 3 digits) stands for direction and number of steps.
        - If the number is in [0, WAREHOUSE_SIZE - 2]: the agent don't make any change for number + 1 times.
        - If the number is in [WAREHOUSE_SIZE - 1, 2 * WAREHOUSE_SIZE - 3]: the agent go up for number + 1 times.
        - If the number is in [2 * WAREHOUSE_SIZE - 2, 3 * WAREHOUSE_SIZE - 4]: the agent go down for number + 1 times.
        - If the number is in [3 * WAREHOUSE_SIZE - 3, 4 * WAREHOUSE_SIZE - 5]: the agent go left for number + 1 times.
        - If the number is in [4 * WAREHOUSE_SIZE - 4, 5 * WAREHOUSE_SIZE - 6]: the agent go right for number + 1 times.
    """

    @staticmethod
    def encode(route: np.ndarray, warehouse: Warehouse = warehouse_model) -> str:
        """
        :param route:
        :param warehouse:
        :return:
        """
        # Remove -1:
        encoded = route[np.where(route != -1)]
        start, end = encoded[1], encoded[-1]

        encoded = encoded.reshape(-1, 2)
        directions = []
        for i, _ in enumerate(encoded):
            if not i:
                continue
            if encoded[i][0] == encoded[i - 1][0] + 1:
                directions.append(Moves.UP)
            elif encoded[i][0] == encoded[i - 1][0] - 1:
                directions.append(Moves.DOWN)
            elif encoded[i][1] == encoded[i - 1][1] - 1:
                directions.append(Moves.LEFT)
            elif encoded[i][1] == encoded[i - 1][1] + 1:
                directions.append(Moves.RIGHT)
            elif encoded[i][0] == encoded[i - 1][0] and encoded[i][1] == encoded[i - 1][1]:
                directions.append(Moves.STAY)
            else:
                raise ValueError('route has an illegal move!')

        count = 1
        zipped_directions = []
        for i, _ in enumerate(directions):
            if i == 0:
                continue
            if directions[i] == directions[i - 1]:
                count += 1
            else:
                zipped_directions.extend([directions[i - 1], count])
                count = 1
            if i == len(directions) - 1:
                zipped_directions.extend([directions[i], count])

        full_directions = np.asarray(zipped_directions).reshape(-1, 2)
        embedded = [str(direction) + '0' * (2 - len(str(num_steps))) + str(num_steps) for direction, num_steps in full_directions]
        return f'({start},{end})_{"".join(embedded)}'

    def create_encodings_from_routes(self, directory_src_path: str, target_path: str):
        encodings = []
        for sub_dir in os.listdir(directory_src_path):
            print(sub_dir)
            src_dst_directory = os.path.join(directory_src_path, sub_dir)
            for csv_file in os.listdir(src_dst_directory):
                csv_file = os.path.join(src_dst_directory, csv_file)
                df = pd.read_csv(csv_file).fillna(-1)
                routes = df.values
                encodings.extend([self.encode(route) for route in routes])

        print('Writing encodings')
        filename = os.path.join(target_path, "encoding.txt")
        with open(filename, "w") as outfile:
            outfile.write("\n".join(encodings))

        return len(encodings)

    @staticmethod
    def _resume_route(route, move, num_steps):
        last_cell = route[-1].reshape(-1, 2)

        if move == Moves.UP:
            direction_to_add = np.asarray([1, 0]).reshape(last_cell.shape)
        elif move == Moves.DOWN:
            direction_to_add = np.asarray([-1, 0]).reshape(last_cell.shape)
        elif move == Moves.LEFT:
            direction_to_add = np.asarray([0, -1]).reshape(last_cell.shape)
        elif move == Moves.RIGHT:
            direction_to_add = np.asarray([0, 1]).reshape(last_cell.shape)
        else:
            direction_to_add = np.zeros_like(last_cell)

        for step in range(num_steps):
            route = np.concatenate([route, last_cell + direction_to_add])
            last_cell = route[-1].reshape(-1, 2)

        return route

    def decode(self, encoded):
        wh_size = warehouse_model.length
        encoded_str = str(encoded)
        routing_request, encoded_str = encoded_str.split('_')
        start, end = eval(routing_request)

        embedded_nums = [num for num in textwrap.wrap(encoded_str, 3)]
        route = np.array([[wh_size - 1, start]])
        directions = [(int(num[0]), int(num[1:])) for num in embedded_nums]

        # directions = np.asarray([[num // (wh_size - 1), num % (wh_size - 1) + 1] for num in embedded_nums])
        # start, directions, end = embedded[0], embedded[1:-1], embedded[-1]
        # route = np.array([[wh_size - 1, start]])
        # directions = np.asarray(directions).reshape(-1, 2)
        for direction, num_steps in directions:
            route = self._resume_route(route, direction, num_steps)

        return route


class RouteTokenizer(Tokenizer):
    def __init__(self, num_words=None):
        super().__init__(num_words=num_words, filters='', lower=False)

    def fit_on_texts(self, filepath: str):
        """

        :param filepath: path to route encodings
        :return:
        """
        with open(filepath, 'r') as f:
            examples = f.read().splitlines()
        super().fit_on_texts(texts=examples)

    def write_json_file(self, target_path: str):
        json_tokenizer = super().to_json()
        with open(target_path, 'w') as f:
            f.write(json_tokenizer)


def get_tokenizer_from_json(file: str):
    with open(file, 'r') as f:
        tokenizer_dict = json.load(f)
    return tokenizer_from_json(json.dumps(tokenizer_dict))


def request_to_routes(request, data_dir, max_paths: Optional[int] = None) -> np.ndarray:
    """
    takes a single routing request and turns it into an array of possible routes.
    :param request:
    :param max_paths:
    :return:
    """
    x_start, x_end, arrival_time = request['x_start'], request['x_end'], request['arrival_time']
    routes_file = f'{data_dir}/src_{x_start}_dst_{x_end}/path_length_{arrival_time}.csv'
    df = pd.read_csv(routes_file)
    if max_paths and len(df) > max_paths:
        df = df.head(max_paths)
    return df.values


def encode_pipeline(
        df: pd.DataFrame,
        batch,
        batch_size,
        num_requests: int,
        data_dir: str,
        tokenizer: RouteTokenizer,
        max_paths: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    # Getting the correct batch:
    sliced_df = df[batch * num_requests: (batch + batch_size) * num_requests]
    # This is now a list of paths for each routing request.
    list_of_routes = requests_to_routes(sliced_df, data_dir, max_paths)
    route_encoder = RouteEncoder()
    encoded_routes = []
    for route_list in list_of_routes:
        encoded_routes.append([route_encoder.encode(route) for route in route_list])
    tokenized_routes = tokenizer.texts_to_sequences(encoded_routes)
    for routes in tokenized_routes:
        routes += [0] * (max_paths - len(routes))
    tokenized_routes = [tokenized_routes[i: (i + 1) * num_requests] for i in range(0, len(tokenized_routes), num_requests)]
    tokenized_routes = [list(itertools.chain.from_iterable(l)) for l in tokenized_routes]
    routing_requests = sliced_df.values.tolist()
    routing_requests = [routing_requests[i: (i + 1) * num_requests] for i in range(0, len(routing_requests), num_requests)]
    routing_requests = [list(itertools.chain.from_iterable(l)) for l in routing_requests]
    # converting to array of arrays:
    return np.vstack(tokenized_routes), np.vstack(routing_requests)


def decode_pipeline(tokenized_routes, tokenizer: RouteTokenizer):
    de_tokenized = tokenizer.sequences_to_texts(tokenized_routes)
    route_encoder = RouteEncoder()
    return [route_encoder.decode(encoded) for encoded in de_tokenized]


def requests_to_routes(requests: pd.DataFrame, data_dir, max_paths: Optional[int] = None) -> List[np.ndarray]:
    return [request_to_routes(request, data_dir, max_paths) for _, request in requests.iterrows()]


# Example:
# if __name__ == '__main__':
#     request = {
#         "x_start": 2,
#         "y_start": 9,
#         "x_end": 0,
#         "y_end": 0,
#         "arrival_time": 15,
#     }
#     df = pd.DataFrame([request])
#     data_dir = '../../data_new'
#     routes_dir = f'{data_dir}/routes_by_request'
#     tokenizer = get_tokenizer_from_json(f'{data_dir}/tokenizer.json')
#     result = encode_pipeline(df, 0, 1, routes_dir, tokenizer)
#     result_2 = decode_pipeline(result[0], tokenizer)
