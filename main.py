import os
import csv
import math
import matplotlib.pyplot as plt

from vs.constants import VS
import sequencer


def read_map_local(map_file):
    victims_list = []
    with open(map_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            x = int(row[0])
            y = int(row[1])
            id = int(row[3])
            tri = int(row[16])
            sobr = float(row[17])
            if id != -1:
                victims_list.append((id, x, y, tri, sobr))
    return victims_list


def read_env_config(path):
    cfg = {}
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == 'BASE':
                coords = parts[1].split(',')
                cfg['BASE'] = (int(coords[0]), int(coords[1]))
            elif key == 'GRID_WIDTH':
                cfg['GRID_WIDTH'] = int(parts[1])
            elif key == 'GRID_HEIGHT':
                cfg['GRID_HEIGHT'] = int(parts[1])
    return cfg


def read_agent_config(path):
    cfg = {}
    with open(path, 'r') as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == 'NAME':
                cfg['NAME'] = parts[1]
            elif key in ('COST_LINE', 'COST_DIAG', 'COST_READ', 'COST_FIRST_AID'):
                cfg[key] = float(parts[1])
            elif key == 'TLIM':
                cfg['TLIM'] = float(parts[1])
            elif key == 'COLOR':
                r = int(parts[1].strip('(),'))
                g = int(parts[2].strip('(),'))
                b = int(parts[3].strip('(),'))
                cfg['COLOR'] = (r, g, b)
    return cfg


def load_clusters(cluster_dir):
    files = sorted([f for f in os.listdir(cluster_dir) if f.startswith('cluster_') and f.endswith('.txt')])
    clusters = []
    for fname in files:
        path = os.path.join(cluster_dir, fname)
        with open(path, 'r') as fh:
            members = [int(line.strip()) for line in fh if line.strip()]
        clusters.append({'file': fname, 'members': members})
    return clusters


def compute_centroid(members, victims_pos):
    xs, ys = [], []
    for id in members:
        coord = victims_pos.get(id)
        if coord:
            xs.append(coord[0])
            ys.append(coord[1])
    if not xs:
        return (0.0, 0.0)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def dist(a, b=(0, 0)):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def assign_clusters_greedy(clusters, victims_pos, rescuers, base=(0, 0)):
    # initial potential costs
    pot = {r['NAME']: 0.0 for r in rescuers}
    assigned = {r['NAME']: [] for r in rescuers}

    for i, cl in enumerate(clusters, start=1):
        members = cl['members']
        centroid = compute_centroid(members, victims_pos)
        # potential: number of victims (primary) + 0.5 * distance (secondary, reduced weight)
        cluster_potential = len(members) + 0.5 * dist(centroid, base)
        # find rescuer with lowest potential
        best = min(pot.items(), key=lambda x: x[1])[0]
        assigned[best].append({'index': i, 'members': members, 'centroid': centroid, 'potential': cluster_potential})
        pot[best] += cluster_potential
    return assigned, pot


def main():
    # read environment config for base and grid size
    env_config_path = os.path.join('datasets', 'env', '94x94_408v', 'env_config.txt')
    env_config = read_env_config(env_config_path)
    grid_width = env_config.get('GRID_WIDTH', 94)
    grid_height = env_config.get('GRID_HEIGHT', 94)
    
    # convert base from grid coordinates [0, width-1] to relative coordinates [-(width/2), (width/2)]
    base_grid = env_config.get('BASE', (grid_width // 2, grid_height // 2))
    center = (grid_width - 1) / 2.0
    base = (base_grid[0] - center, base_grid[1] - center)

    map_file = os.path.join('datasets', 'map', 'map.csv')
    victims_list = read_map_local(map_file)
    victims_pos = {v[0]: (v[1], v[2]) for v in victims_list}

    cluster_dir = os.path.join('.', 'clusters')
    clusters = load_clusters(cluster_dir)

    # read rescuer configs
    cfg_folder = os.path.join('.', 'cfg')
    rescuer_cfgs = []
    for i in range(1, 4):
        cfg_path = os.path.join(cfg_folder, f'soc_{i}.txt')
        rescuer_cfgs.append(read_agent_config(cfg_path))

    assigned, potentials = assign_clusters_greedy(clusters, victims_pos, rescuer_cfgs, base=base)

    os.makedirs('outputs', exist_ok=True)

    results = {}

    colors = [(1,0,0), (0,0.5,0), (0,0,1)]

    print(f"Base (grid): {base_grid}")
    print(f"Base (relative): {base}")
    print()

    plt.figure(figsize=(8,8))
    plt.axis('equal')

    # plot obstacles and victims
    obst_x = []
    obst_y = []
    vic_x = []
    vic_y = []
    vic_tri = []
    # read map.csv to get obstacles
    with open(map_file, 'r') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            x = int(row[0])
            y = int(row[1])
            obst = float(row[2])
            id = int(row[3])
            if obst == VS.OBST_WALL or obst >= VS.OBST_WALL:
                obst_x.append(x)
                obst_y.append(y)
            if id != -1:
                vic_x.append(x)
                vic_y.append(y)
                tri = int(row[16])
                vic_tri.append(tri)

    plt.scatter(obst_x, obst_y, c='black', s=4, label='walls')
    plt.scatter(vic_x, vic_y, c='orange', s=6, label='victims')
    plt.scatter(base[0], base[1], c='red', s=80, marker='s', label='base')

    # For each rescuer, build all victims list from assigned clusters and sequence them
    for idx, rcfg in enumerate(rescuer_cfgs):
        name = rcfg.get('NAME', f'SOC_{idx+1}')
        cost_line = rcfg.get('COST_LINE', 1.0)
        cost_diag = rcfg.get('COST_DIAG', 1.5)
        assigned_clusters = assigned.get(name, [])
        all_members = []
        for cl in assigned_clusters:
            all_members.extend(cl['members'])

        # remove duplicates while preserving order
        seen = set()
        members_ordered = [x for x in all_members if not (x in seen or seen.add(x))]

        # sequence all members using sequencer
        route = sequencer.sequence_cluster(members_ordered, victims_pos, base=base, iterations=3000, temp0=1.0, cooling=0.995,
                                           cost_line=cost_line, cost_diag=cost_diag)
        total_cost = sequencer.route_cost(route, victims_pos, base=base, cost_line=cost_line, cost_diag=cost_diag)

        results[name] = {
            'assigned_clusters': [c['index'] for c in assigned_clusters],
            'route': route,
            'cost': total_cost
        }

        print(f"\n{name}")
        print(f"- Assigned clusters: {results[name]['assigned_clusters']}")
        print(f"- Cost: {total_cost:.2f}")

        # save route
        out_route = os.path.join('outputs', f'route_{name}.txt')
        with open(out_route, 'w') as fh:
            fh.write('\n'.join(str(v) for v in route))

        # draw route on plot
        xs = [base[0]]
        ys = [base[1]]
        for id in route:
            coord = victims_pos.get(id, base)
            xs.append(coord[0])
            ys.append(coord[1])

        plt.plot(xs, ys, color=colors[idx % len(colors)], linewidth=1.5, label=f'route_{name}')

    plt.legend()
    plt.title('Rescuers routes and map')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == '__main__':
    main()
