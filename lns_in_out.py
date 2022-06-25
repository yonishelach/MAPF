import csv

HEIGHT = 32
WIDTH = 32
IN_FILE = 'in.scen'
OUT_FILE = 'out'


def get_lns_paths():
    with open(OUT_FILE) as out:
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


def get_lns_demands():
    with open(IN_FILE) as out:
        lines = out.readlines()[1:]
    out.close()
    demands = []
    for line in lines:
        line = line.split('\t')
        demands.append((int(line[4]), int(line[5])))
    return demands


def fill_paths():
    lns_paths = get_lns_paths()
    lns_demands = get_lns_demands()
    for i in range(len(lns_paths)):
        path = [lns_demands[i][0], lns_demands[i][1]]
        for j in range(1, len(lns_paths[i])):
            if lns_paths[i][j] == lns_paths[i][j-1]:
                path += [path[-2], path[-1]]
            elif lns_paths[i][j] == lns_paths[i][j-1] + HEIGHT:
                path += [path[-2], path[-1]+1]
            elif lns_paths[i][j] == lns_paths[i][j-1] - HEIGHT:
                path += [path[-2], path[-1]-1]
            elif lns_paths[i][j] == lns_paths[i][j-1] + 1:
                path += [path[-2]+1, path[-1]]
            elif lns_paths[i][j] == lns_paths[i][j-1] - 1:
                path += [path[-2]-1, path[-1]]
        paths.append(path)


def get_max_path_length():
    result = 0
    for path in paths:
        result = max(result, len(path))
    return result


if __name__ == '__main__':
    paths = []
    fill_paths()
    header = ['y_0']
    for j in range(1, get_max_path_length()):
        if header[-1][0] == 'y':
            header.append('x_' + header[-1][2])
        else:
            header.append('y_' + str(int(header[-1][2])+1))

    with open('./output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(paths)
        f.close()
