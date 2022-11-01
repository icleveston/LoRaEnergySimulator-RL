import os
import pickle
from GlobalConfig import locations_file
from Framework.Location import Location

num_locations = 50
cell_size = 1000
num_of_simulations = 1

locations_per_simulation = list()

for num_sim in range(num_of_simulations):
    locations = list()
    for i in range(num_locations):
        locations.append(Location(min=0, max=cell_size, indoor=False))
    locations_per_simulation.append(locations)


os.makedirs(os.path.dirname(locations_file), exist_ok=True)
with open(locations_file, 'wb') as f:
    pickle.dump(locations_per_simulation, f)

