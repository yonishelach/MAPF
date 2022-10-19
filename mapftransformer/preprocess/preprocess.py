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
    - Each direction: stay, up, down, left and right has a matching integer as you can see in Moves.
    - A route is encoded from an array of coordinates into a string of integers of the form '(src, dst)_<a_1><b_1>...<a_n><b_n>' in the following way:
        - src is the y's source coordinate.
        - dst is the y's destination coordinate.
        - Each <a_i> is the move matching integer.
        - Each <b_i> is the number of steps to move in <a_i> direction.
    """

    @staticmethod
    def encode(route: np.ndarray) -> str:
        """
        :param route:   A numpy array that represents coordinates of the routes.

        :return:    A string representation of the encoded route as described in the class.
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
        """
        Encode routes from a selected directory source path and creates a text file with all the encodings, line by line.
        The source path suppose to be the one that contains the output of `database.routes.create_csv_files()`.
        :param directory_src_path:  The directory path that contains all the csv files. The routes that are inside will be encoded.
        :param target_path:         The directory to store the text file. The filename will be "encodings.txt".

        :return:    The number of routes that were encoded.
        """
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
        """
        Appending to the route the next coordinates based on the route.
        :param route:       The route to continue.
        :param move:        The direction to go.
        :param num_steps:   The number of steps to go in that direction.

        :return:    The updated route.
        """
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
        """
        Decoding back the encoded route to a numpy array of coordinates.
        :param encoded: The encoded route.
        :return:    The route by coordinates in the warehouse.
        """
        wh_size = warehouse_model.length
        encoded_str = str(encoded)
        routing_request, encoded_str = encoded_str.split('_')
        start, end = eval(routing_request)

        embedded_nums = [num for num in textwrap.wrap(encoded_str, 3)]
        route = np.array([[wh_size - 1, start]])
        directions = [(int(num[0]), int(num[1:])) for num in embedded_nums]

        for direction, num_steps in directions:
            route = self._resume_route(route, direction, num_steps)

        return route


class RouteTokenizer(Tokenizer):
    def __init__(self, num_words=None):
        super().__init__(num_words=num_words, filters='', lower=False)

    def fit_on_texts(self, filepath: str):
        """
        given a text file of routes, fitting the tokenizer on it.

        :param filepath: path to route encodings file.
        :return:    None.
        """
        with open(filepath, 'r') as f:
            examples = f.read().splitlines()
        super().fit_on_texts(texts=examples)

    def write_json_file(self, target_path: str):
        """
        Writing the tokenizer to a json file. In this way you can save your tokenizer and not fit it every time.
        :param target_path: The filename ta save the tokenizer json file.
        :return:    None.
        """
        json_tokenizer = super().to_json()
        with open(target_path, 'w') as f:
            f.write(json_tokenizer)


def get_tokenizer_from_json(file: str):
    """
    Loading a fitted tokenizer from a json file.
    :param file:    The json file to of the fitted tokenizer.
    :return:    The fitted tokenizer object.
    """
    with open(file, 'r') as f:
        tokenizer_dict = json.load(f)
    return tokenizer_from_json(json.dumps(tokenizer_dict))


def request_to_routes(request, data_dir, max_routes: Optional[int] = None) -> np.ndarray:
    """
    Takes a single routing request and turns it into an array of possible routes.

    :param request:     A pandas DataFrame / Series with one entry that has the following headers: 'x_start', 'x_end' and 'arrival_time'.
    :param data_dir:    The directory that possess all the routes (in CSV files).
    :param max_routes:  The maximum number of routes to take from the CSV file.
    :return:    A numpy array of matching routes to the routing request.
    """
    x_start, x_end, arrival_time = request['x_start'], request['x_end'], request['arrival_time']
    routes_file = f'{data_dir}/src_{x_start}_dst_{x_end}/path_length_{arrival_time}.csv'
    df = pd.read_csv(routes_file)
    if max_routes and len(df) > max_routes:
        df = df.head(max_routes)
    return df.values


def encode_pipeline(
        df: pd.DataFrame,
        batch,
        batch_size,
        num_requests: int,
        data_dir: str,
        tokenizer: RouteTokenizer,
        max_routes: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A pipeline that converts a batch of routing requests to a numpy array of tokenized routes.

    :param df:              The DataFrame that contains the routing requests.
    :param batch:           The batch number.
    :param batch_size:      The batch size.
    :param num_requests:    The number of requests in each routing request.
    :param data_dir:        The data directory name.
    :param tokenizer:       The matching fitted tokenizer.
    :param max_routes:      The maximum number of routes per request.
    :return:    A tuple of two numpy arrays where the first one is the tokenized routes, and the second one is of the routing requests themselves.
    """
    # Getting the correct batch:
    sliced_df = df[batch * num_requests: (batch + batch_size) * num_requests]
    # This is now a list of paths for each routing request.
    list_of_routes = requests_to_routes(sliced_df, data_dir, max_routes)
    route_encoder = RouteEncoder()
    encoded_routes = []
    for route_list in list_of_routes:
        encoded_routes.append([route_encoder.encode(route) for route in route_list])
    tokenized_routes = tokenizer.texts_to_sequences(encoded_routes)
    for routes in tokenized_routes:
        routes += [0] * (max_routes - len(routes))
    tokenized_routes = [tokenized_routes[i: (i + 1) * num_requests] for i in range(0, len(tokenized_routes), num_requests)]
    tokenized_routes = [list(itertools.chain.from_iterable(l)) for l in tokenized_routes]
    routing_requests = sliced_df.values.tolist()
    routing_requests = [routing_requests[i: (i + 1) * num_requests] for i in range(0, len(routing_requests), num_requests)]
    routing_requests = [list(itertools.chain.from_iterable(l)) for l in routing_requests]
    # converting to array of arrays:
    return np.vstack(tokenized_routes), np.vstack(routing_requests)


def decode_pipeline(tokenized_routes, tokenizer: RouteTokenizer):
    """
    Takes tokenized routes and turn them into arrays of coordinates.
    :param tokenized_routes:    A list of tokenized routes.
    :param tokenizer:           The matching tokenizer.
    :return:    A list of routes in the form of arrays with coordinates.
    """
    de_tokenized = tokenizer.sequences_to_texts(tokenized_routes)
    route_encoder = RouteEncoder()
    return [route_encoder.decode(encoded) for encoded in de_tokenized]


def requests_to_routes(requests: pd.DataFrame, data_dir, max_paths: Optional[int] = None) -> List[np.ndarray]:
    """
    Takes a bunch of routing requests and generate a list of possible routes to every routing request.
    :param requests:    The DataFrame with the routing requests.
    :param data_dir:    The directory with all the routes.
    :param max_paths:   The maximum number of routes per request to generate.
    :return:    A list of routes.
    """
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
