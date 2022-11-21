import random
import gym
from gym import spaces
import numpy as np
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
from Framework.Location import Location
from Framework.LoRaParameters import LoRaParameters
from Framework.Node import Node
from Framework.SNRModel import SNRModel


class LoraEnvironment(gym.Env):

    def __init__(self, locations_file='locations/1_12000.pkl', adr=False, confirmed_messages=True, printed=False):
        super(LoraEnvironment, self).__init__()

        self.locations_file = locations_file
        self.adr = adr
        self.confirmed_messages = confirmed_messages
        self.printed = printed

        self.iteration = 0
        self.payload_size = None
        self.bw = None
        self.tp = None
        self.location = None
        self.gateway_location = Location(x=0, y=0, indoor=False)

        scaling_factor = 0.1

        # 12*8 bits per hour (1 typical packet per hour)
        self.transmission_rate = scaling_factor * (12 * 8) / (60 * 60 * 1000)
        self.simulation_time = 24 * 60 * 60 * 1000 / scaling_factor
        self.sigma = 20  # [0, 5, 7.8, 15, 20]
        self.MAX_DELAY_BEFORE_SLEEP_MS = 500
        self.MAX_DELAY_START_PER_NODE_MS = np.round(self.simulation_time / 10)

        # Create the action and observation space
        self.action_space = spaces.Discrete(12,)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]),
                                             high=np.array([1, 1, 1, 1]),
                                             shape=(4,), dtype=np.int)

    def _create(self, sf, tp, payload_size, location):

        env = simpy.Environment()

        gateway = Gateway(env, self.gateway_location, max_snr_adr=False, avg_snr_adr=False, printed=self.printed)

        air_interface = AirInterface(gateway, PropagationModel.LogShadow(std=self.sigma), SNRModel(), env, self.printed)

        nodes = []

        tx_power_mW = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}
        rx_measurements = {'pre_mW': 8.2, 'pre_ms': 3.4, 'rx_lna_on_mW': 39,
                           'rx_lna_off_mW': 34, 'post_mW': 8.3, 'post_ms': 10.7}

        for node_id in range(1):
            energy_profile = EnergyProfile(5.7e-3, 15, tx_power_mW, rx_power=rx_measurements)

            lora_param = LoRaParameters(freq=np.random.choice(LoRaParameters.DEFAULT_CHANNELS),
                                        sf=sf,
                                        bw=self.bw,
                                        cr=5,
                                        crc_enabled=1,
                                        de_enabled=0,
                                        header_implicit_mode=0,
                                        tp=tp)

            node = Node(node_id,
                        energy_profile,
                        lora_param,
                        sleep_time=(8 * self.payload_size / self.transmission_rate),
                        process_time=5,
                        adr=False,
                        location=location,
                        base_station=gateway,
                        env=env,
                        payload_size=payload_size,
                        air_interface=air_interface,
                        MAX_DELAY_START_PER_NODE_MS=self.MAX_DELAY_START_PER_NODE_MS,
                        MAX_DELAY_BEFORE_SLEEP_MS=self.MAX_DELAY_BEFORE_SLEEP_MS,
                        confirmed_messages=self.confirmed_messages,
                        printed=False)

            nodes.append(node)
            env.process(node.run())

        self.env, self.nodes, self.gateway, self.air_interface = env, nodes, gateway, air_interface

    def reset(self):

        random.seed(1)

        self.iteration = 0
        self.payload_size = 50
        self.bw = 125
        self.tp = 2
        self.location = Location(5000, 5000)

        obs = np.array([
            self.payload_size/100,
            5000/5000,
            5000/5000,
            int(Location.distance(self.gateway_location, self.location))/7500
        ])

        return obs

    def _next_observation(self):

        self.payload_size = random.randint(1, 100)
        self.location = Location(self.location.x - random.randint(0, 20), self.location.y - random.randint(0, 20))

        obs = np.array([
            self.payload_size/100,
            self.location.x/5000,
            self.location.y/5000,
            int(Location.distance(self.gateway.location, self.location))/7500
        ])

        return obs

    def _take_action(self, action):

        if action == 0:
            sf = 7
            tp = 2
        elif action == 1:
            sf = 8
            tp = 2
        elif action == 2:
            sf = 9
            tp = 2
        elif action == 3:
            sf = 10
            tp = 2
        elif action == 4:
            sf = 11
            tp = 2
        elif action == 5:
            sf = 12
            tp = 2
        elif action == 6:
            sf = 7
            tp = 14
        elif action == 7:
            sf = 8
            tp = 14
        elif action == 8:
            sf = 9
            tp = 14
        elif action == 9:
            sf = 10
            tp = 14
        elif action == 10:
            sf = 11
            tp = 14
        elif action == 11:
            sf = 12
            tp = 14

        self._create(sf=sf, tp=tp, payload_size=self.payload_size, location=self.location)

    def step(self, action):

        self._take_action(action)

        self.env.run()

        reward = 1/(1+self.nodes[0].num_retransmission) + 1/(1+self.nodes[0].energy_per_bit())

        obs = self._next_observation()

        done = self.iteration > 1000

        self.iteration += 1

        return obs, reward, done, {}

    def render(self):
        self.gateway.log()

    def get_stats(self):

        mean_energy_per_bit_list = []

        for n in self.nodes:
            mean_energy_per_bit_list.append(n.energy_per_bit())

        data_mean_nodes = Node.get_mean_simulation_data_frame(self.nodes, name=self.sigma) / self.num_nodes

        data_gateway = self.gateway.get_simulation_data() / self.num_nodes
        data_air_interface = self.air_interface.get_simulation_data() / self.num_nodes

        results = {
            'mean_nodes': data_mean_nodes,
            'gateway': data_gateway,
            'air_interface': data_air_interface,
            'path_loss_std': self.sigma,
            'payload_size': self.payload_size,
            'mean_energy_all_nodes': mean_energy_per_bit_list,
            'cell_size': self.area_size,
            'adr': self.adr,
            'confirmed_messages': self.confirmed_messages,
            'total_devices': self.num_nodes,
            'transmission_rate': self.transmission_rate,
            'simulation_time': self.simulation_time,
            'num_of_simulations_done': 0,
        }

        return results

        #results_file = f"results/{self.num_nodes}_{datetime.datetime.now()}.pkl"

        #os.makedirs(os.path.dirname(results_file), exist_ok=True)

        #pickle.dump(results, open(results_file, "wb"))


if __name__ == '__main__':

    env = LoraEnvironment()


