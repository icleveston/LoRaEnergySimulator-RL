import datetime
import os
import pickle
import numpy as np
import simpy
from Framework import Location as loc
from Framework import PropagationModel
from Framework.AirInterface import AirInterface
from Framework.EnergyProfile import EnergyProfile
from Framework.Gateway import Gateway
from Framework.LoRaParameters import LoRaParameters
from Framework.Node import Node
from Framework.SNRModel import SNRModel

scaling_factor = 0.1
transmission_rate_id = str(scaling_factor)
transmission_rate_bit_per_ms = scaling_factor * (12 * 8) / (
        60 * 60 * 1000)  # 12*8 bits per hour (1 typical packet per hour)
simulation_time = 24 * 60 * 60 * 1000 / scaling_factor
payload_sizes = range(5, 55, 5)
path_loss_variances = 7.9  # [0, 5, 7.8, 15, 20]
MAX_DELAY_BEFORE_SLEEP_MS = 500
MAX_DELAY_START_PER_NODE_MS = np.round(simulation_time / 10)


def main(locations_file, payload_size=50, adr=True, confirmed_messages=True):

    with open(locations_file, 'rb') as file_handler:
        data = pickle.load(file_handler)
        locations = data['locations']
        area_size = data['area_size']
        num_nodes = len(locations)

    env, nodes, gateway, air_interface = create(locations,
                                                p_size=payload_size,
                                                sigma=path_loss_variances,
                                                area_size=area_size,
                                                transmission_rate=transmission_rate_bit_per_ms,
                                                confirmed_messages=confirmed_messages,
                                                adr=adr)

    state = None
    while True:

        # Take action


        if env.peek() < simulation_time:
            env.step()

        # Get new state
        state = gateway
        #gateway.log()

    # Process data
    mean_energy_per_bit_list = list()
    for n in nodes:
        mean_energy_per_bit_list.append(n.energy_per_bit())

    data_mean_nodes = Node.get_mean_simulation_data_frame(nodes, name=path_loss_variances) / num_nodes

    data_gateway = gateway.get_simulation_data(name=path_loss_variances) / num_nodes
    data_air_interface = air_interface.get_simulation_data(name=path_loss_variances) / num_nodes

    results = {
        'mean_nodes': data_mean_nodes,
        'gateway': data_gateway,
        'air_interface': data_air_interface,
        'path_loss_std': path_loss_variances,
        'payload_size': payload_size,
        'mean_energy_all_nodes': mean_energy_per_bit_list,
        'cell_size': area_size,
        'adr': adr,
        'confirmed_messages': confirmed_messages,
        'total_devices': num_nodes,
        'transmission_rate': transmission_rate_bit_per_ms,
        'simulation_time': simulation_time,
        'num_of_simulations_done': 0,
    }

    results_file = f"results/{num_nodes}_{datetime.datetime.now()}.pkl"

    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    pickle.dump(results, open(results_file, "wb"))


def create(locs, p_size, sigma, area_size, transmission_rate, confirmed_messages, adr):

    env = simpy.Environment()

    gateway_location = loc.Location(x=area_size // 2, y=area_size // 2, indoor=False)

    gateway = Gateway(env, gateway_location, max_snr_adr=True, avg_snr_adr=False)

    air_interface = AirInterface(gateway, PropagationModel.LogShadow(std=sigma), SNRModel(), env)

    nodes = []

    tx_power_mW = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}
    rx_measurements = {'pre_mW': 8.2, 'pre_ms': 3.4, 'rx_lna_on_mW': 39,
                       'rx_lna_off_mW': 34, 'post_mW': 8.3, 'post_ms': 10.7}

    for node_id in range(len(locs)):

        energy_profile = EnergyProfile(5.7e-3, 15, tx_power_mW, rx_power=rx_measurements)

        lora_param = LoRaParameters(freq=np.random.choice(LoRaParameters.DEFAULT_CHANNELS),
                                    sf=np.random.choice(LoRaParameters.SPREADING_FACTORS),
                                    bw=125,
                                    cr=5,
                                    crc_enabled=1,
                                    de_enabled=0,
                                    header_implicit_mode=0,
                                    tp=14)

        node = Node(node_id,
                    energy_profile,
                    lora_param,
                    sleep_time=(8 * p_size / transmission_rate),
                    process_time=5,
                    adr=adr,
                    location=locs[node_id],
                    base_station=gateway,
                    env=env,
                    payload_size=p_size,
                    air_interface=air_interface,
                    MAX_DELAY_START_PER_NODE_MS=MAX_DELAY_START_PER_NODE_MS,
                    MAX_DELAY_BEFORE_SLEEP_MS=MAX_DELAY_BEFORE_SLEEP_MS,
                    confirmed_messages=confirmed_messages)

        nodes.append(node)
        env.process(node.run())

    return env, nodes, gateway, air_interface


if __name__ == '__main__':
    main(locations_file='locations/100_12000.pkl')
