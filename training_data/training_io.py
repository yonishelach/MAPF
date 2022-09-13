import csv
import random
from solution_validator import solution_validator
from constants import *
from lns_io import create_lns_input_files, create_lns_output_files


def get_entries(num):
    result = []
    while len(result) < num:
        result.append(random.randrange(WIDTH))
        result = list(dict.fromkeys(result))
    return result


def create_requests_batches_without_time():
    routing_requests_batches = []
    sources = get_entries(NUMBER_OF_SOURCES)
    targets = get_entries(NUMBER_OF_TARGETS)
    for i in range(NUMBER_OF_TRAINING_BATCHES):
        batch = []
        for j in range(LNS_BATCH_SIZE):
            batch.append([sources[random.randrange(NUMBER_OF_SOURCES)], WIDTH-1,
                          targets[random.randrange(NUMBER_OF_TARGETS)], 0])
        routing_requests_batches.append(batch)
    return routing_requests_batches


def get_requests(batch_index: int):
    with open(LNS_REQUESTS_DIRECTORY + '/batch_' + str(batch_index) + '.scen') as f:
        lines = f.readlines()[1:]
    f.close()
    requests = []
    for line in lines:
        line = line.split('\t')
        requests.append((int(line[4]), int(line[5])))
    return requests


def get_requests_batches_without_time():
    requests_batches = []
    for i in range(NUMBER_OF_TRAINING_BATCHES):
        requests_batches.append(get_requests(i))
    return requests_batches


def get_lns_solution(batch_index: int):
    """
    :param batch_index: The index for a batch of an output file.
    :return: A lists of paths were each path is represented by a list of numbers (int).
    The numbers represent a location in the warehouse in lns format:
    A number -n- represent the location (x,y) were: n = x + y * WIDTH.
    """
    with open('lns_outputs/batch_' + str(batch_index)) as f:
        lines = f.readlines()
        f.close()
    index = 0
    lns_solution = []
    for line in lines:
        line = line.split(' ')
        if line[0] == "Agent" and line[1] == str(index) + ':':
            index += 1
            line = line[2].split('\t')[:-1]
            lns_solution.append([int(x) for x in line])
    return lns_solution


def get_solution(batch_index: int):
    """
    :param batch_index: The index for a batch of lns paths from get_lns_paths.
    :return: A list of paths were each path is represented by a list of numbers x1,y1,x2,y2,x3,y3,...
    such that (xi, yi) is a location in the warehouse.
    """
    solution = []
    lns_paths = get_lns_solution(batch_index)
    lns_requests = requests_batches_without_time[batch_index]
    for i in range(len(lns_paths)):
        path = [lns_requests[i][0], lns_requests[i][1]]
        for j in range(1, len(lns_paths[i])):
            if lns_paths[i][j] == lns_paths[i][j - 1]:
                path += [path[-2], path[-1]]
            elif lns_paths[i][j] == lns_paths[i][j - 1] + WIDTH:
                path += [path[-2], path[-1] + 1]
            elif lns_paths[i][j] == lns_paths[i][j - 1] - WIDTH:
                path += [path[-2], path[-1] - 1]
            elif lns_paths[i][j] == lns_paths[i][j - 1] + 1:
                path += [path[-2] + 1, path[-1]]
            elif lns_paths[i][j] == lns_paths[i][j - 1] - 1:
                path += [path[-2] - 1, path[-1]]
        solution.append(path)
    return solution


def get_training_solutions():
    training_solutions = [get_solution(batch_index) for batch_index in range(NUMBER_OF_TRAINING_BATCHES)]
    training_solutions = [solution_validator(solution) for solution in training_solutions]
    return training_solutions


def request_by_path(path: list):
    result = path[:2] + path[-2:]
    result.append(int(len(path)/2))
    result = [str(x) for x in result]
    result = ','.join(result) + '\n'
    return result


def get_training_requests_batches():
    requests_batches = []
    training_solutions = get_training_solutions()
    print(training_solutions[0])
    for solution in training_solutions:
        requests_batches.append([request_by_path(path) for path in solution])
    return requests_batches


def create_training_requests_file():
    requests_batches = get_training_requests_batches()
    with open('./training_requests.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(requests_batches)):
            writer.writerow(['Batch_' + str(i)])
            writer.writerows([request[:-1].split(',') for request in requests_batches[i]])
        f.close()


if __name__ == '__main__':
    requests_batches_without_time = create_requests_batches_without_time()
    print(requests_batches_without_time)
    create_lns_input_files(requests_batches_without_time)
    create_lns_output_files(requests_batches_without_time)
    # requests_batches_without_time = get_requests_batches_without_time()
    create_training_requests_file()
