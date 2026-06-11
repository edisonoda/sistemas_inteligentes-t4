import math
import random
from typing import List, Dict, Tuple


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def move_cost(a: Tuple[int, int], b: Tuple[int, int], cost_line: float, cost_diag: float) -> float:
    dx = abs(int(a[0]) - int(b[0]))
    dy = abs(int(a[1]) - int(b[1]))
    diag = min(dx, dy)
    straight = max(dx, dy) - diag
    return diag * cost_diag + straight * cost_line


def route_cost(route: List[int], pos: Dict[int, Tuple[float, float]], base=(0, 0),
               cost_line: float = 1.0, cost_diag: float = 1.5) -> float:
    if not route:
        return 0.0
    cost = 0.0
    prev = base
    for id in route:
        coord = pos.get(id)
        if coord is None:
            # if missing position, add a large penalty
            cost += 1e6
        else:
            # use grid movement cost when available
            cost += move_cost(prev, coord, cost_line, cost_diag)
            prev = coord
    return cost


def sequence_cluster(cluster_ids: List[int], victims_pos: Dict[int, Tuple[float, float]],
                     base=(0, 0), iterations=2000, temp0=1.0, cooling=0.995,
                     cost_line: float = 1.0, cost_diag: float = 1.5) -> List[int]:
    """Simple simulated annealing to order victim IDs minimizing Euclidean travel
    starting at `base`. Returns an ordered list of victim ids.
    """
    if not cluster_ids:
        return []

    # initial solution: greedy nearest neighbor from base
    remaining = set(cluster_ids)
    curr = base
    route = []
    while remaining:
        best = None
        best_d = float('inf')
        for id in list(remaining):
            coord = victims_pos.get(id)
            if coord is None:
                d = float('inf')
            else:
                d = move_cost(curr, coord, cost_line, cost_diag)
            if d < best_d:
                best_d = d
                best = id
        route.append(best)
        remaining.remove(best)
        curr = victims_pos.get(best, curr)

    best_route = route[:]
    best_cost = route_cost(best_route, victims_pos, base, cost_line, cost_diag)

    curr_route = best_route[:]
    curr_cost = best_cost
    T = temp0

    for it in range(iterations):
        # neighbor: swap two positions
        i, j = random.sample(range(len(curr_route)), 2)
        new_route = curr_route[:]
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_cost = route_cost(new_route, victims_pos, base, cost_line, cost_diag)

        delta = new_cost - curr_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-8)):
            curr_route = new_route
            curr_cost = new_cost
            if curr_cost < best_cost:
                best_route = curr_route[:]
                best_cost = curr_cost
        T *= cooling
        if T < 1e-6:
            break

    return best_route
