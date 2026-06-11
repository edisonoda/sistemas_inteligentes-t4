##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### Not a complete version of DFS; it comes back prematuraly
### to the base when it enters into a dead end position


import csv
import glob
import heapq
import math
import os
import re

from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from sequencer import sequence_cluster


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.map = Map()            # only SOC_1 has all maps (it is the master)
        self.victims = {}           # list of found victims
        self.plan = []              # a list of planned actions
        self.plan_x = 0             # the x position of the rescuer during the planning phase
        self.plan_y = 0             # the y position of the rescuer during the planning phase
        self.plan_visited = set()   # positions already planned to be visited 
        self.plan_rtime = self.TLIM # the remaing time during the planning phase
        self.plan_walk_time = 0.0   # previewed time to walk during rescue
        self.x = 0                  # the current x position of the rescuer when executing the plan
        self.y = 0                  # the current y position of the rescuer when executing the plan
        self.explorers_remaining = {"EXP_1", "EXP_2", "EXP_3"} # control explorers
        self.rescuers = []          # list of all rescuers
        self.assigned_clusters = []  # clusters assigned to this rescuer
        self.potential_cost = 0.0    # assigned cluster cost accumulator
                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    def set_rescuers(self, rescuers_lst):
        """ each rescuer has the reference to the others"""
        self.rescuers = rescuers_lst
        
    def do_rescue(self, map, clusters):
        """ O agente socorrista executa a estratégia de salvamento tendo
            o mapa e os clusters que foram atribuídos a ele.
        """
        self.set_state(VS.ACTIVE)

        if map:
            self._merge_map_data(map)

        self._update_victims_from_map()
        self.assigned_clusters = clusters or []
        self.plan = []
        self.plan_visited = set()
        self.plan_rtime = self.TLIM
        self.plan_walk_time = 0.0

        base = tuple(self.get_env().dic["BASE"])
        self.plan_x, self.plan_y = base
        self.x, self.y = base

        if not self.assigned_clusters:
            print(f"{self.NAME}: no clusters assigned to this rescuer")
            return

        victims_pos = self._victim_positions()
        ordered_victim_ids = []

        for cluster in self.assigned_clusters:
            ordered = sequence_cluster(
                self.get_env(),
                cluster["ids"],
                victims_pos,
                base=base,
                cost_line=self.COST_LINE,
                cost_diag=self.COST_DIAG
            )
            cluster["ids"] = ordered
            ordered_victim_ids.extend(ordered)
            self._save_ordered_cluster(cluster["file"], ordered)

        self.plan = self._build_plan_from_targets(ordered_victim_ids, victims_pos, base)

        if not self.plan:
            print(f"{self.NAME}: no valid rescue plan could be built")
        else:
            print(f"{self.NAME}: plan ready with {len(self.plan)} steps")

    def merge_maps(self, exp_name, map, victims):
        """ The explorer named exp_name sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""

        self._merge_map_data(map)
        print(f"{self.NAME}: Map received from explorer {exp_name}")

        self.victims.update(victims)
        self.explorers_remaining.discard(exp_name)

        if self.explorers_remaining:
            print(f"{self.NAME}: Waiting for remaining explorers... {self.explorers_remaining}")
            return

        self._update_victims_from_map()
        self.map.draw()

        clusters = self._read_clusters_folder()
        base = tuple(self.get_env().dic["BASE"])
        victims_pos = self._victim_positions()

        clusters_sorted = sorted(
            clusters,
            key=lambda item: self._cluster_cost(item["ids"], victims_pos, base),
            reverse=True
        )

        for cluster in clusters_sorted:
            best_rescuer = min(self.rescuers, key=lambda r: getattr(r, "potential_cost", 0.0))
            cluster_cost = self._cluster_cost(cluster["ids"], victims_pos, base)
            best_rescuer.assigned_clusters.append(cluster)
            best_rescuer.potential_cost += cluster_cost
            print(f"{self.NAME}: assigned {cluster['file']} to {best_rescuer.NAME} with cost {cluster_cost:.2f}")

        for rescuer in self.rescuers:
            rescuer.do_rescue(self.map, rescuer.assigned_clusters)
            
        
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           print(f"{self.NAME} has finished the plan")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy, there_is_vict = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} vict: {there_is_vict}")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")
            # check if there is a victim at the current position
            if there_is_vict:
                rescued = self.first_aid() # True when rescued
                if rescued:
                    print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                else:
                    print(f"{self.NAME} Plan fail - victim not found at ({self.x}, {self.x})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        #input(f"{self.NAME} remaining time: {self.get_rtime()} Tecle enter")

        return True

    def _merge_map_data(self, source_map):
        for coord, cell_data in source_map.map_data.items():
            if not self.map.in_map(coord):
                difficulty, victim_seq, actions_res = cell_data
                self.map.add(coord, difficulty, victim_seq, actions_res)

    def _victim_positions(self):
        positions = {}
        for coord, cell_data in self.map.map_data.items():
            _, victim_seq, _ = cell_data
            if victim_seq != VS.NO_VICTIM:
                positions[victim_seq] = coord

        for vid, coord in enumerate(self.get_env().victims):
            positions.setdefault(vid, coord)

        return positions

    def _update_victims_from_map(self):
        updated = {}
        for coord, cell_data in self.map.map_data.items():
            _, victim_seq, _ = cell_data
            if victim_seq != VS.NO_VICTIM:
                existing = self.victims.get(victim_seq)
                vitals = existing[1] if existing else None
                updated[victim_seq] = (coord, vitals)

        if not updated:
            for vid, coord in enumerate(self.get_env().victims):
                updated[vid] = (coord, None)

        self.victims = updated

    def _read_clusters_folder(self):
        clusters_dir = os.path.join(os.path.dirname(__file__), "clusters")
        clusters = []
        if not os.path.isdir(clusters_dir):
            return clusters

        paths = sorted(
            glob.glob(os.path.join(clusters_dir, "cluster_*.txt")),
            key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1))
        )

        for path in paths:
            with open(path, "r") as file:
                ids = [int(line.strip()) for line in file if line.strip()]
                clusters.append({"file": os.path.basename(path), "ids": ids})

        return clusters

    def _cluster_cost(self, ids, positions, base):
        if not ids:
            return 0.0

        centroid = self._centroid(ids, positions)
        distance = math.hypot(centroid[0] - base[0], centroid[1] - base[1])
        return len(ids) + 0.5 * distance

    def _centroid(self, ids, positions):
        cells = [positions.get(v_id) for v_id in ids if positions.get(v_id) is not None]
        if not cells:
            return (0.0, 0.0)

        x_sum = sum(c[0] for c in cells)
        y_sum = sum(c[1] for c in cells)
        return (x_sum / len(cells), y_sum / len(cells))

    def _save_ordered_cluster(self, cluster_name, ordered_ids):
        out_dir = os.path.join(os.path.dirname(__file__), "ordered_clusters")
        os.makedirs(out_dir, exist_ok=True)

        cluster_base = os.path.splitext(cluster_name)[0]
        path = os.path.join(out_dir, f"{cluster_base}_{self.NAME}.txt")

        with open(path, "w") as file:
            for victim_id in ordered_ids:
                file.write(f"{victim_id}\n")

    def _heuristic(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return self.COST_DIAG * min(dx, dy) + self.COST_LINE * abs(dx - dy)

    def _neighbours(self, coord):
        env = self.get_env()
        width = env.dic["GRID_WIDTH"]
        height = env.dic["GRID_HEIGHT"]
        for dx, dy in AbstAgent.AC_INCR.values():
            nx = coord[0] + dx
            ny = coord[1] + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if env.obst[nx][ny] == VS.OBST_WALL:
                continue
            yield (nx, ny), dx, dy

    def _move_cost(self, from_coord, to_coord):
        env = self.get_env()
        target_cost = env.obst[to_coord[0]][to_coord[1]]
        dx = abs(from_coord[0] - to_coord[0])
        dy = abs(from_coord[1] - to_coord[1])
        diag = min(dx, dy)
        straight = max(dx, dy) - diag
        base_cost = diag * self.COST_DIAG + straight * self.COST_LINE
        return base_cost * target_cost

    def _astar_path(self, start, goal):
        if start == goal:
            return []

        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, goal), start))
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                return list(reversed(path))

            for neighbour, _, _ in self._neighbours(current):
                tentative_g = g_score[current] + self._move_cost(current, neighbour)
                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    f_score[neighbour] = tentative_g + self._heuristic(neighbour, goal)
                    heapq.heappush(open_set, (f_score[neighbour], neighbour))

        return []

    def _build_plan_from_targets(self, target_ids, positions, base):
        plan = []
        current = base
        handled_coords = set()
        coord_to_victim = {coord: vid for vid, coord in positions.items()}

        for vid in target_ids:
            target = positions.get(vid)
            if target is None:
                print(f"{self.NAME}: victim {vid} has no known position")
                continue

            path = self._astar_path(current, target)
            if not path and current != target:
                print(f"{self.NAME}: no path to victim {vid} at {target}")
                continue

            for step in path:
                dx = step[0] - current[0]
                dy = step[1] - current[1]
                there_is_vict = 1 if step in coord_to_victim and step not in handled_coords else 0
                if there_is_vict:
                    handled_coords.add(step)
                plan.append((dx, dy, there_is_vict))
                current = step

        if current != base:
            return_path = self._astar_path(current, base)
            for step in return_path:
                dx = step[0] - current[0]
                dy = step[1] - current[1]
                plan.append((dx, dy, 0))
                current = step

        return plan
