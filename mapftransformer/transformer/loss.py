import tensorflow as tf
import numpy as np


def count_collisions(plan: np.ndarray):
    """
    Given a plan of routes, count the number of collisions between all the planned routes of two types: vertex conflict and swapping conflict.
    - Vertex conflict is when two agents are in the same coordinates at the same time.
      Each two agents that participate in such a conflict counted as one collision.
    - Swapping conflict is when two agents are switching places (coordinates) in the same time.
      Each two agents that participate in such a conflict counted as one collision.

    :param plan:    The plan of the routes.

    :return:    The total number of collisions.
    """
    # positions = np.zeros((*WAREHOUSE_SIZE, MAX_LEN), dtype=int)
    positions = {}
    # in the same place at the same time:
    num_vertex_conflict = 0
    # switching places (the same "edge" at the same time):
    num_swapping_conflicts = 0

    # Changing shape to
    # plan = plan.reshape((NUM_RR, -1, 2))

    for agent in range(len(plan)):

        path = plan[agent]
        prev_location = path[0]

        for time in range(len(path)):
            location = path[time]
            if all(location == path[0]) or all(location == path[-1]) or all(location == -1):  # agent is at source or at destination
                continue

            # add position to visit map:
            prev_pos_key = f'{prev_location[0]}_{location[1]}'
            pos_key = f'{location[0]}_{location[1]}'
            if pos_key not in positions:
                positions[pos_key] = {}
            if time not in positions[pos_key]:
                positions[pos_key][time] = []
            else:
                num_vertex_conflict += len(positions[pos_key][time])

            # find swapping conflicts:
            if all(prev_location != path[0]):
                if (time in positions[prev_pos_key]) and (time - 1 in positions[pos_key]):
                    for other_agent in positions[pos_key][time - 1]:
                        if other_agent in positions[prev_pos_key][time]:
                            num_swapping_conflicts += 1

            positions[pos_key][time].append(agent)
            prev_location = location
    return num_vertex_conflict + num_swapping_conflicts


def validate_routing_request(prediction: np.ndarray, routing_request: np.ndarray):
    """
    Checks if all the routing requests are fulfilled by the prediction (contains routes) and counts the number of misses.
    :param prediction:      The array of routes.
    :param routing_request: The routing requests.
    :return:    The number of misses.
    """
    rr_dict = {}
    for x_start, _, x_end, _, arriving_time in routing_request:
        rr_dict[f'{x_start}_{x_end}_{arriving_time}'] = 1

    for path in prediction:
        src = path[0][1]
        dst = path[np.argwhere(path == [-1, -1])[0][0] - 1][1]
        arriving_time = len(path)
        rr_dict[f'{src}_{dst}_{arriving_time}'] -= 1

    return sum(abs(v) for v in rr_dict.values()) // 2


def loss_fn(prediction: np.ndarray, routing_request: np.ndarray, w_l: float = 2.0) -> tf.Tensor:
    """
    The goal of the loss function is to check if there are no collisions in the plan, and that each routing request has a path.
    Therefore, it counts the collisions in the plan and the number of misses (routing requests that are not fulfilled) and sum them.

    :param prediction:      The plan of routes that predicted in the transformer.
    :param routing_request: The routing request that is relevant to the prediction.
    :param w_l:             The weight for the number of misses, defaulted to 2.0.

    Returns:    A tensor of loss result.
    """

    # Count collisions:
    num_collisions = count_collisions(prediction)
    # Validate all routing requests were fulfilled:
    num_misses = validate_routing_request(prediction, routing_request)
    return tf.constant(num_collisions + w_l * num_misses)
