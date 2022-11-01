import os
import pickle
from Framework.Location import Location


def main(num_nodes, area_size):

    locations_file = f"locations/{num_nodes}_{area_size}.pkl"

    locations = []
    for i in range(num_nodes):
        locations.append(Location(min=0, max=area_size, indoor=False))

    os.makedirs(os.path.dirname(locations_file), exist_ok=True)

    with open(locations_file, 'wb') as f:
        pickle.dump({'locations': locations, 'num_nodes': num_nodes, 'area_size': area_size}, f)


if __name__ == '__main__':

    main(num_nodes=100, area_size=12000)
