import csv
import random
from solution_validator import is_solution_valid
from config import *
from lns_io import create_lns_input_files, create_lns_output_files
from training_data.debugging_utils import get_requests_batches_without_time


def create_requests_without_time_batch():
    x_start_list = [i for i in range(NUMBER_OF_SOURCES)]
    x_end_list = [i for i in range(NUMBER_OF_TARGETS)]
    random.shuffle(x_end_list)
    batch = []
    for i in range(BATCH_SIZE):
        request = [random.choice(x_start_list) * int(WAREHOUSE_WIDTH/NUMBER_OF_SOURCES),
                   WAREHOUSE_WIDTH - 1,
                   x_end_list[i] * int(WAREHOUSE_WIDTH/NUMBER_OF_TARGETS),
                   0]
        batch.append(request)
    return batch


def create_requests_batches_without_time():
    """
    Creates a list of random requests batches without time.
    :return: The requests batches.
    """
    requests_batches = [create_requests_without_time_batch() for _ in range(NUMBER_OF_TRAINING_BATCHES)]
    return requests_batches


def get_output_file_lines(batch_index: int):
    with open(LNS_OUTPUTS_DIRECTORY + '/batch_' + str(batch_index)) as f:
        lines = f.readlines()
    f.close()
    return lines


def is_path_line(line: str):
    return line[0] == "Agent" and line[1].endswith(':')


def read_lns_solution(batch_index: int):
    """
    Read a solution from the lns output file for a given batch id
    :param batch_index: The index for a batch of an output file.
    :return: A lists of paths were each path is represented by a list of numbers (int).
    The numbers represent a location in the warehouse in lns format:
    A number -n- represent the location (x,y) were: n = x + y * WAREHOUSE_WIDTH.
    """
    lines = get_output_file_lines(batch_index)
    lines = [line.split(' ') for line in lines]
    lines = filter(lambda line: is_path_line(line), lines)
    lines = [line[2].split('\t')[:-1] for line in lines]
    lns_solution = [[int(x) for x in line] for line in lines]
    return lns_solution


def get_coordinates_by_y(lns_position_value: int, y: int):
    return [lns_position_value - (y * WAREHOUSE_WIDTH), y]


def get_lns_position_by_coordinates(coordinates: [int, int]):
    return coordinates[0] + (coordinates[1] * WAREHOUSE_WIDTH)


def get_coordinates_by_prev(lns_position_value: int, prev_coordinates: [int, int]):
    turns = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for turn in turns:
        coordinates = [prev_coordinates[0] + turn[0], prev_coordinates[1] + turn[1]]
        if lns_position_value == get_lns_position_by_coordinates(coordinates):
            return coordinates
    return prev_coordinates


def convert_lns_path_to_transformer_format(lns_path):
    path = [get_coordinates_by_y(lns_path[0], WAREHOUSE_WIDTH - 1)]
    for lns_pos in lns_path[1:]:
        coordinates = get_coordinates_by_prev(lns_pos, path[-1])
        path.append(coordinates)
    return path


def convert_lns_solution_to_transformer_format(lns_solution):
    solution = [convert_lns_path_to_transformer_format(lns_path) for lns_path in lns_solution]
    return solution


def get_valid_solutions():
    valid_solutions = []
    for batch_index in range(NUMBER_OF_TRAINING_BATCHES):
        lns_solution = read_lns_solution(batch_index)
        solution = convert_lns_solution_to_transformer_format(lns_solution)
        if is_solution_valid(solution):
            valid_solutions.append(solution)
    return valid_solutions


def request_by_path(path: list):
    result = path[0] + path[-1]
    result.append(len(path) + 1)
    result = [str(x) for x in result]
    result = ','.join(result) + '\n'
    return result


def get_training_requests_batches():
    requests_batches = []
    training_solutions = get_valid_solutions()
    for solution in training_solutions:
        requests_batches.append([request_by_path(path) for path in solution])
    return requests_batches


def create_training_data_file():
    """
    The final stage - using the solutions from the lns algorithm to create
    the requests batches in the transformer format.
    :return: A csv file named training_data.csv with the requests batches in the transformer format.
    """
    requests_batches = get_training_requests_batches()
    with open('./training_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x_start", "y_start", "x_end", "y_end", "arrival_time"])
        for i in range(len(requests_batches)):
            writer.writerows([request[:-1].split(',') for request in requests_batches[i]])
        f.close()


if __name__ == '__main__':
    requests_batches_without_time = create_requests_batches_without_time()
    create_lns_input_files(requests_batches_without_time)
    create_lns_output_files(requests_batches_without_time)
    # requests_batches_without_time = get_requests_batches_without_time()
    create_training_data_file()
