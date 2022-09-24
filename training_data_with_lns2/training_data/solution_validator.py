from itertools import tee

from config import *
from typing import List


def is_valid_path_length(path):
    return len(path) in range(MIN_PATH_LENGTH, MAX_PATH_LENGTH)


def is_valid_path_start_end_points(path):
    return path[0][1] == WAREHOUSE_WIDTH - 1 and path[-1][1] == 0


def is_valid_path_steps(path):
    current_step = path[0]
    for i in range(1, len(path)):
        next_step = path[i]
        if current_step[0]-next_step[0] not in [-1, 0, 1]:
            return False
        if current_step[1]-next_step[1] not in [-1, 0, 1]:
            return False
        if current_step[0] != next_step[0] and current_step[1] != next_step[1]:
            return False
        if current_step[1] < next_step[1]:
            return False
        current_step = next_step
    return True


def is_valid_path(path: List[int]):
    result = is_valid_path_length(path)
    result = result and is_valid_path_start_end_points(path)
    result = result and is_valid_path_steps(path)
    return result


def remove_bad_paths(solution: List[List[int]]):
    return list(filter(lambda path: is_valid_path(path), solution))


def is_collision_exists(path1, path2):
    current_step1 = path1[0]
    current_step2 = path2[0]
    for i in range(1, min(len(path1), len(path2))):
        next_step1 = path1[i]
        next_step2 = path2[i]
        if next_step1 == next_step2 and next_step1[1] not in [0, WAREHOUSE_WIDTH-1]:
            return True
        if current_step1[1] == current_step2[1]:
            if next_step1[0] == current_step2[0] and next_step2[0] == current_step1[0]:
                return True
        current_step1 = next_step1
        current_step2 = next_step2
    return False


def find_collision(solution):
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            if is_collision_exists(solution[i], solution[j]):
                return j
    return -1


def remove_collisions(solution):
    index = find_collision(solution)
    while index != -1:
        del solution[index]
        index = find_collision(solution)
    return solution


def is_solution_valid(solution):
    """
    Checking a solution
    :param solution: A solution in the transformer format for a requests batch.
    :return: True if the solution is a valid solution and False otherwise.
    """
    solution = remove_bad_paths(solution)
    solution = remove_collisions(solution)
    return len(solution) == BATCH_SIZE
