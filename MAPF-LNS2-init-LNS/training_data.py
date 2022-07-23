import csv
import os
import threading
import random
from numpy.linalg import norm

HEIGHT = 40
WIDTH = 40
TRANSFORMER_BATCH_SIZE = 30
LNS_BATCH_SIZE = 60
MAX_PATH_LENGTH = 200
LNS_REQUESTS_FILE = 'custom_lns_requests.scen'
# 'custom_lns_requests.scen'
# 'random-32-32-20-random-12.scen'
LNS_OUTPUT_DIRECTORY = 'training_data/lns_outputs'
LNS_REQUESTS_DIRECTORY = 'training_data/lns_requests'
ROUTING_REQUESTS_FILE = 'database/routing_requests_without_time.csv'
LNS_RESULT_PATHS_FILE = 'lns_result_paths.csv'
MAP_FILE = 'random-40-40-20.map'


# returns a list of pairs (start, target)
def get_all_routing_requests():
    with open(ROUTING_REQUESTS_FILE) as f:
        lines = f.readlines()[1:]
    f.close()
    routing_requests = [line.split(',')[:-1] for line in lines]
    routing_requests = [req for requests in routing_requests for req in requests]
    result = []
    for i in range(0, len(routing_requests), 2):
        result.append((int(routing_requests[i]), int(routing_requests[i+1])))
    return result


def create_initial_requests_batches():
    routing_requests = get_all_routing_requests()
    routing_requests_batches = []
    for i in range(0, len(routing_requests)-LNS_BATCH_SIZE+1, LNS_BATCH_SIZE):
        routing_requests_batches.append(routing_requests[i:i+LNS_BATCH_SIZE])
    return routing_requests_batches


def create_lns_request_line(request: (int, int)):
    return str(round(random.uniform(0, 20))) + "\trandom-32-32-20.map\t40\t40\t" + \
           str(request[0]) + "\t39\t" + str(request[1]) + "\t0\t" + \
           str(norm([request[0] - request[1], 39], 2)) + "\n"


def create_lns_requests_files():
    for i in range(len(initial_requests_batches)):
        rows = ["version 1\n"]
        rows += [create_lns_request_line(request) for request in initial_requests_batches[i]]
        with open('./training_data/lns_requests/batch_' + str(i) + '.scen', 'w', newline='') as f:
            f.writelines(rows)
            f.close()


def create_lns_output_file(index: str):
    command_start = './lns -m ' + MAP_FILE + ' -a training_data/lns_requests/batch_'
    command_middle = '.scen -o test.csv -k 51 -t 100 -s 3 > training_data/lns_outputs/batch_'
    os.system(command_start + index + command_middle + index)


def create_lns_output_files():
    threads = []
    for i in range(len(initial_requests_batches)):
        x = threading.Thread(target=create_lns_output_file, args=(str(i),))
        threads.append(x)
        x.start()
        if i > 0 and i % 100 == 0:
            for thread in threads:
                thread.join()
            threads = []


def get_lns_paths(batch_id: int):
    with open(LNS_OUTPUT_DIRECTORY + '/batch_' + str(batch_id)) as f:
        lines = f.readlines()
    f.close()
    index = 0
    lns_paths = []
    for line in lines:
        line = line.split(' ')
        if line[0] == "Agent" and line[1] == str(index) + ':':
            index += 1
        else:
            continue
        line = line[2].split('\t')[:-1]
        lns_paths.append([int(x) for x in line])
    return lns_paths


def get_lns_requests(batch_id: int):
    with open(LNS_REQUESTS_DIRECTORY + '/batch_' + str(batch_id) + '.scen') as f:
        lines = f.readlines()[1:]
    f.close()
    requests = []
    for line in lines:
        line = line.split('\t')
        requests.append((int(line[4]), int(line[5])))
    return requests


def get_paths(batch_id: int):
    paths = []
    lns_paths = get_lns_paths(batch_id)
    lns_requests = get_lns_requests(batch_id)
    for i in range(len(lns_paths)):
        path = [lns_requests[i][0], lns_requests[i][1]]
        for j in range(1, len(lns_paths[i])):
            if lns_paths[i][j] == lns_paths[i][j - 1]:
                path += [path[-2], path[-1]]
            elif lns_paths[i][j] == lns_paths[i][j - 1] + HEIGHT:
                path += [path[-2], path[-1] + 1]
            elif lns_paths[i][j] == lns_paths[i][j - 1] - HEIGHT:
                path += [path[-2], path[-1] - 1]
            elif lns_paths[i][j] == lns_paths[i][j - 1] + 1:
                path += [path[-2] + 1, path[-1]]
            elif lns_paths[i][j] == lns_paths[i][j - 1] - 1:
                path += [path[-2] - 1, path[-1]]
        paths.append(path)
    return paths


def is_valid_path(path):
    if len(path) < 4 or len(path) > MAX_PATH_LENGTH:
        return False
    for i in range(1, len(path)-3, 2):
        if path[i] < path[i+2]:
            return False
    return True


def remove_bad_paths(paths):
    return list(filter(lambda path: is_valid_path(path), paths))


def does_collision_exists(path1, path2):
    parallel_time = int(min(len(path1), len(path2))/2)
    for i in range(1, parallel_time):
        x = (i-1)*2
        y = x + 1
        if path1[y] == 0 or path1[y] == WIDTH-1 or path2[y] == 0 or path2[y] == WIDTH-1:
            continue
        if path1[x] == path2[x] and path1[y] == path2[y]:
            # print("collision!")
            # print("i = " + str(i))
            return True
    return False


def does_collision_exists_in_paths(paths):
    for i in range(len(paths)-1):
        for j in range(i+1, len(paths)):
            if does_collision_exists(paths[i], paths[j]):
                return True
    return False


def remove_collision(paths):
    for i in range(len(paths)-1):
        for j in range(i+1, len(paths)):
            if does_collision_exists(paths[i], paths[j]):
                del paths[j]
                return paths
    return paths


def remove_collisions(paths):
    while does_collision_exists_in_paths(paths):
        paths = remove_collision(paths)
    return paths


def get_solution(batch_id):
    paths = remove_collisions(remove_bad_paths(get_paths(batch_id)))
    paths = paths[:min(len(paths), TRANSFORMER_BATCH_SIZE)]
    return paths


def get_training_paths_batches():
    paths_batches = [get_solution(batch_id) for batch_id in range(len(initial_requests_batches))]
    return paths_batches


def create_training_solutions_file():
    num_columns = 0
    for batch in training_paths_batches:
        for path in batch:
            num_columns = max(num_columns, len(path))

    header = ['x_'+str(i) + ',y_' + str(i) for i in range(int(num_columns/2))]
    header = ','.join(header).split(',')

    with open('./training_data/training_solutions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(training_paths_batches)):
            writer.writerow(['Batch_' + str(i)])
            writer.writerows(training_paths_batches[i])
        f.close()


def request_by_path(path: list):
    result = path[:2] + path[-2:]
    result.append(int(len(path)/2))
    result = [str(x) for x in result]
    result = ','.join(result) + '\n'
    return result


def get_training_requests_batches():
    requests_batches = []
    for paths_batch in training_paths_batches:
        requests_batches.append([request_by_path(path) for path in paths_batch])
    return requests_batches


def create_training_requests_file():
    requests_batches = get_training_requests_batches()
    with open('./training_data/training_requests.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(requests_batches)):
            writer.writerow(['Batch_' + str(i)])
            writer.writerows([request[:-1].split(',') for request in requests_batches[i]])
        f.close()


if __name__ == '__main__':
    initial_requests_batches = create_initial_requests_batches()
    # create_lns_requests_files()
    # create_lns_output_files()
    training_paths_batches = get_training_paths_batches()
    # create_training_solutions_file()
    create_training_requests_file()
