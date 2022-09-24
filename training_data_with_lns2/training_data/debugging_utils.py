from config import *


def get_input_file_lines(batch_index: int):
    with open(LNS_INPUTS_DIRECTORY + '/batch_' + str(batch_index) + '.scen') as f:
        lines = f.readlines()[1:]
    f.close()
    return lines


def get_requests_without_time(input_file_lines):
    input_file_lines = [line.split('\t') for line in input_file_lines]
    requests = [[int(line[4]), WAREHOUSE_WIDTH - 1, int(line[5]), 0] for line in input_file_lines]
    return requests


def get_requests_batches_without_time():
    """
    read from the lns input files and returns requests batches.
    :return: A list of requests batches without time in the transformer format
    """
    requests_batches = []
    for i in range(NUMBER_OF_TRAINING_BATCHES):
        batch = get_requests_without_time(get_input_file_lines(i))
        requests_batches.append(batch)
    return requests_batches
