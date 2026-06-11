import math
import random
from typing import List, Dict, Tuple


# Global obstacle map - will be loaded from file
OBSTACLE_MAP = {}


def load_obstacles(obst_file: str):
    """Load obstacle costs from file. Format: x,y,cost"""
    global OBSTACLE_MAP
    OBSTACLE_MAP.clear()
    with open(obst_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                x, y, cost = int(parts[0]), int(parts[1]), float(parts[2])
                OBSTACLE_MAP[(x, y)] = cost


def get_cell_cost(x: int, y: int) -> float:
    """Get cost multiplier for cell (x,y). Default is 1.0 if no obstacle."""
    if (x, y) in OBSTACLE_MAP:
        return OBSTACLE_MAP[(x, y)]
    return 1.0


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def move_cost(a: Tuple[int, int], b: Tuple[int, int], cost_line: float, cost_diag: float) -> float:
    """Calculate movement cost from a to b, accounting for obstacle costs along the path."""
    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    diag = min(dx, dy)
    straight = max(dx, dy) - diag
    
    # Base movement cost (no obstacles)
    base_cost = diag * cost_diag + straight * cost_line
    
    # Add obstacle costs along the path by sampling waypoints
    if OBSTACLE_MAP:
        # Use Bresenham-like line sampling to get cells along the path
        cells_on_path = []
        steps = max(dx, dy) + 1
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = int(round(a[0] + t * (b[0] - a[0])))
            y = int(round(a[1] + t * (b[1] - a[1])))
            cells_on_path.append((x, y))
        
        # Sum up obstacle costs for cells on path
        obst_cost = 0.0
        for cell in cells_on_path:
            cell_cost = get_cell_cost(cell[0], cell[1])
            # Only add extra cost if cell has obstacle (cost > 1.0)
            if cell_cost > 1.0:
                obst_cost += (cell_cost - 1.0) * 0.1  # Scale down obstacle cost to avoid dominating movement
        
        return base_cost + obst_cost
    
    return base_cost


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
