##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### Not a complete version of DFS; it comes back prematuraly
### to the base when it enters into a dead end position


from vs.constants import VS

# Sequencing helper (simulated annealing implementation)
import glob
import os
import sequencer


class Rescuer():
    def __init__(self, config_file):
        # Specific initialization for the rescuer
        self.victims = {}           # list of found victims
        self.x = 0                  # the current x position of the rescuer when executing the plan
        self.y = 0                  # the current y position of the rescuer when executing the plan
        self.rescuers = []          # list of all rescuersself.NAME = ""     # public: the name of the agent
        self.TLIM = 0.0    # public: time limit to execute (cannot be exceeded)
        self.COST_LINE = 0.0        # public: basic cost to walk one step hor or vertically
        self.COST_DIAG = 0.0        # public: basic cost to walk one step diagonally
        self.COST_READ = 0.0        # public: basic cost to read a victim's vital sign
        self.COST_FIRST_AID = 0.0   # public: basic cost to drop the first aid package to a victim
        self.COLOR = (100, 100, 100)       # public: color of the agent
        self.TRACE_COLOR = (140, 140, 140) # public: color for the visited cells
        # stores the folder where the agents' config files are
        self.config_folder = os.path.dirname(config_file)     

        # Read agents config file for controlling time
        with open(config_file, "r") as file:

            # Read each line of the file
            for line in file:
                # Split the line into words
                words = line.split()

                # Get the keyword and value
                keyword = words[0]
                if keyword == "NAME":
                    self.NAME = words[1]
                elif keyword == "COLOR":
                    r = int(words[1].strip('(), '))
                    g = int(words[2].strip('(), '))
                    b = int(words[3].strip('(), '))
                    self.COLOR = (r, g, b)  # a tuple
                elif keyword == "TRACE_COLOR":
                    r = int(words[1].strip('(), '))
                    g = int(words[2].strip('(), '))
                    b = int(words[3].strip('(), '))
                    self.TRACE_COLOR = (r, g, b)  # a tuple
                elif keyword == "TLIM":
                    self.TLIM = float(words[1])
                elif keyword == "COST_LINE":
                    self.COST_LINE = float(words[1])
                elif keyword == "COST_DIAG":
                    self.COST_DIAG = float(words[1])
                elif keyword == "COST_FIRST_AID":
                    self.COST_FIRST_AID = float(words[1])
                elif keyword == "COST_READ":
                    self.COST_READ = float(words[1])

    def set_rescuers(self, rescuers_lst):
        """ each rescuer has the reference to the others"""
        self.rescuers = rescuers_lst
        
    def do_rescue(self, map, clusters):
        # Load victim positions from map.csv (produced by explorers)
        map_file = os.path.join('.', 'map.csv')
        victims_list = []
        if os.path.exists(map_file):
            # import read_map lazily to avoid importing heavy libs at module import time
            from t3.main import read_map
            victims_list = read_map(map_file)

        victims_pos = {v[0]: (v[1], v[2]) for v in victims_list}

        # Read cluster files and assign clusters to this rescuer by round-robin
        cluster_files = sorted(glob.glob(os.path.join('.', 'clusters', 'cluster_*.txt')))
        if not cluster_files:
            print(f"{self.NAME}: No cluster files found in clusters/ directory.")
            return

        # derive rescuer index from NAME (SOC_1 -> 0)
        try:
            idx = int(self.NAME.split('_')[-1]) - 1
        except Exception:
            idx = 0

        for i, fname in enumerate(cluster_files, start=1):
            # assign cluster i to rescuer idx when (i-1) % 3 == idx
            if (i - 1) % 3 != idx:
                continue

            with open(fname, 'r') as fh:
                cluster_ids = [int(line.strip()) for line in fh if line.strip()]

            if not cluster_ids:
                print(f"{self.NAME}: cluster file {fname} is empty, skipping.")
                continue

            print(f"{self.NAME}: Sequencing cluster {i} with {len(cluster_ids)} victims...")

            ordered = sequencer.sequence_cluster(cluster_ids, victims_pos)

            # Overwrite the cluster file with the ordered victim ids (one per line)
            with open(fname, 'w') as fh:
                for vid in ordered:
                    fh.write(f"{vid}\n")

            print(f"{self.NAME}: Wrote ordered cluster to {fname}")
        