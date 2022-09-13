import os
import threading
from time import sleep

from numpy.linalg import norm
from constants import *


def create_lns_input_line(request: (int, int)):
    """
    :param: request: A pair (start: int, target: int) representing a request without time.
    :return: A string representing the line for the given request in the lns requests file.
    """
    return "0\t" + MAP_FILE + "\t" + str(HEIGHT) + "\t" + str(WIDTH) + "\t" + \
           str(request[0]) + "\t" + str(WIDTH-1) + "\t" + str(request[1]) + "\t0\t" + \
           str(norm([request[0] - request[1], WIDTH-1], 2)) + "\n"


def create_lns_input_files(requests_batches_without_time):
    """
    Creates lns request file for each batch in requests_without_time_batches.
    """
    for i in range(len(requests_batches_without_time)):
        rows = ["version 1\n"]
        rows += [create_lns_input_line(request) for request in requests_batches_without_time[i]]
        with open('./lns_inputs/batch_' + str(i) + '.scen', 'w', newline='') as f:
            f.writelines(rows)
            f.close()


def create_lns_output_file(index: str):
    """
    Creates lns output file for the lns input file with batch index equal to the given index.
    :param index: index of a batch
    """
    command_start = '../lns -m ../' + MAP_FILE + ' -a lns_inputs/batch_'
    command_middle = '.scen -o test.csv -k ' + str(LNS_BATCH_SIZE) + ' -t ' + str(MAX_LNS_RUN_TIME) + \
                     ' -s 3 > lns_outputs/batch_'
    os.system(command_start + index + command_middle + index)


def create_lns_output_files(requests_batches_without_time):
    """
    Creates all the lns output files based on all the lns input files created by previous methods.
    The method is using multi-threading to accelerate calculation time.
    """
    threads = []
    for i in range(len(requests_batches_without_time)):
        x = threading.Thread(target=create_lns_output_file, args=(str(i),))
        threads.append(x)
        x.start()
        if i > 0 and i % NUMBER_OF_THREADS == 0:
            for thread in threads:
                thread.join()
            threads = []
    for thread in threads:
        thread.join()

