def distance(u, v):
    return abs(u[0] - v[0]) + abs(u[1] - v[1])
    # return sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)


def update_plan(plan, i, route):
    if not route:
        return
    for step in route:
        plan[i].append(step)
