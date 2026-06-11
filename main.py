import os

# importa classes
from soc import Rescuer

dic = {}    # dicionário para armazenar as configurações do ambiente

def read_map(map_file):
    obstacles = [[1 for y in range(dic["GRID_HEIGHT"])] for x in range(dic["GRID_WIDTH"])]
    victims_list = []

    with open(map_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Pula a primeira linha (cabeçalho)

        for row in csvreader:
            x = int(row[0])  # coordenada x
            y = int(row[1])  # coordenada y
            obst = float(row[2])
            id = int(row[3])   # victim id number
            tri = int(row[16])  # Triagem START: 0 GRN, 1 YEL, 2 RED, 3 BLK
            sobr = float(row[17])  # Prob. de sobrevivencia

            if id != -1:
                victims_list.append((id, x, y, tri, sobr))
            else:
                if obst > 100:
                    obstacles[x][y] = VS.OBST_WALL   # wall
                elif obst <= 0:
                    obstacles[x][y] = VS.OBST_NONE     # no obstacle
                else:
                    obstacles[x][y] = obst
    
    return victims_list

def main(vict_folder, env_folder, config_ag_folder):
    size_file = os.path.join(env_folder, "env_config.txt")
    with open(size_file, "r") as file:
        # Read each line of the file
        for line in file:
            # Split the line into words
            words = line.split()

            # Get the keyword and value
            keyword = words[0]
            raw_value = words[1]

            # casts the value
            if keyword == "BASE":
                value = [int(i) for i in raw_value.split(',')]
            elif keyword == "DELAY":
                value = float(raw_value)
            else:
                value = int(raw_value)

            dic[keyword] = value

    victims_list = read_map(os.path.join(".", "datasets/map/map.csv"))

    soc = []       # agentes socorristas
    
    for i in range(3):
        soc.append(Rescuer(env, os.path.join(config_ag_folder, f"soc_{i+1}.txt")))
        soc[i].set_rescuers(soc)

if __name__ == '__main__':
    print("------------------")
    print("--- INICIO SMA ---")
    print("------------------")
    # dataset com sinais vitais das vitimas
    grid_str = "94x94"
    vict_str = "408v"
    vict_folder = os.path.join(".", "datasets/vict/", vict_str)

    # dataset do ambiente (paredes, posicao das vitimas)
    env_folder = path = os.path.join(".", "datasets", "env", f"{grid_str}_{vict_str}")

    # folder das configuracoes dos agentes
    curr = os.getcwd()
    config_ag_folder = os.path.join(curr, "cfg")

    main(vict_folder, env_folder, config_ag_folder)
