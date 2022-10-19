from .warehouse import Warehouse

NUM_RR = 8
NUM_PP = 2
NUM_AGENTS = NUM_RR
NUM_RELEVANT = NUM_RR * NUM_PP

warehouse_config = dict(
    warehouse_id=1,
    length=10,
    width=10,
    number_of_sources=5,
    number_of_destinations=5,
    static_obstacle_length=round(0.1 * 10),
    static_obstacle_width=round(0.1 * 10),
    static_obstacle_layout=[],
    is_warehouse_searchable=True
)

warehouse_model = Warehouse(**warehouse_config)

WAREHOUSE_SIZE = (warehouse_model.length, warehouse_model.width)
MAX_LEN = sum(WAREHOUSE_SIZE)


