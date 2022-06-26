import csv
from numpy.linalg import norm

HEIGHT = 32
WIDTH = 32
LNS_REQUESTS_FILE = 'lns_requests.scen'
LNS_PATHS_FILE = 'lns_paths'
ROUTING_REQUESTS_FILE = 'database/routing_requests_without_time.csv'


def get_lns_paths():
    with open(LNS_PATHS_FILE) as out:
        lines = out.readlines()
    out.close()
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


def get_lns_requests():
    with open(LNS_REQUESTS_FILE) as out:
        lines = out.readlines()[1:]
    out.close()
    demands = []
    for line in lines:
        line = line.split('\t')
        demands.append((int(line[4]), int(line[5])))
    return demands


def fill_paths(paths):
    lns_paths = get_lns_paths()
    lns_demands = get_lns_requests()
    for i in range(len(lns_paths)):
        path = [lns_demands[i][0], lns_demands[i][1]]
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


def get_max_path_length(paths):
    result = 0
    for path in paths:
        result = max(result, len(path))
    return result


def make_lns_result_paths_file():
    paths = []
    fill_paths(paths)
    header = ['y_0']
    for j in range(1, get_max_path_length(paths)):
        if header[-1][0] == 'y':
            header.append('x_' + header[-1][2])
        else:
            header.append('y_' + str(int(header[-1][2]) + 1))

    with open('./lns_result_paths.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(paths)
        f.close()


def get_routing_requests():
    with open(ROUTING_REQUESTS_FILE) as f:
        lines = f.readlines()[1:]
    f.close()
    routing_requests = [line.split(',') for line in lines]
    return [req for requests in routing_requests for req in requests]


def create_lns_request_line(src: int, dst: int):
    return "7    random-32-32-20.map    40    40    " + str(src) + "    39    " + str(dst) + "    0    " +\
           str(norm([src - dst, 39], 2)) + "\n"


def make_custom_lns_requests_file():
    routing_requests = get_routing_requests()
    rows = ["version 1\n"]
    for i in range(0, len(routing_requests) - 1, 2):
        rows.append(create_lns_request_line(int(routing_requests[i]), int(routing_requests[i + 1])))
    with open('./custom_lns_requests.scen', 'w', newline='') as f:
        f.writelines(rows)
        f.close()


if __name__ == '__main__':
    make_lns_result_paths_file()
    make_custom_lns_requests_file()
