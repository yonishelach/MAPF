from constants import *
from typing import List


def is_valid_path_length(path):
    return len(path) in range(MIN_PATH_LENGTH, MAX_PATH_LENGTH) and len(path) % 2 == 0


def is_valid_path_start_end_points(path):
    return path[1] == WIDTH - 1 and path[-1] == 0


def is_valid_path_steps(path):
    current_step = path[:2]
    for i in range(2, len(path), 2):
        next_step = path[i:i+2]
        if abs(current_step[0]-next_step[0]) not in [0, 1]:
            return False
        if abs(current_step[1]-next_step[1]) not in [0, 1]:
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
    for path in solution:
        print(path)
        if not is_valid_path_length(path):
            print("DROPPED BY LENGTH")
            print("path length is: " + str(len(path)))
        elif not is_valid_path_start_end_points(path):
            print("DROPPED BY START END POINTS")
        elif not is_valid_path_steps(path):
            print("DROPPED BY PATH STEPS")
    return list(filter(lambda path: is_valid_path(path), solution))


def is_collision_exists(path1, path2):
    current_step1 = path1[:2]
    current_step2 = path2[:2]
    for i in range(2, min(len(path1), len(path2)), 2):
        next_step1 = path1[i:i+2]
        next_step2 = path2[i:i+2]
        if next_step1[1] == WIDTH-1 or next_step2[1] == WIDTH-1:
            current_step1 = next_step1
            current_step2 = next_step2
            continue
        if next_step1[0] == next_step2[0] and next_step1[1] == next_step2[1] and next_step1[1] not in [0, WIDTH-1]:
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
                print("collision between paths:")
                print(solution[i])
                print(solution[j])
                return j
    return -1


def remove_collisions(solution):
    index = find_collision(solution)
    while index != -1:
        del solution[index]
        index = find_collision(solution)
    return solution


def solution_validator(solution: List[List[int]]):
    solution = remove_bad_paths(solution)
    print("AFTER BAD PATHS")
    print(len(solution))
    print("AFTER COLLISIONS")
    solution = remove_collisions(solution)
    print(len(solution))
    if len(solution) < TRANSFORMER_BATCH_SIZE:
        print("ERROR: solution too small")
    solution = solution[:TRANSFORMER_BATCH_SIZE]
    return solution
