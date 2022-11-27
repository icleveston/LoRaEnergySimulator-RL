import random
import matplotlib.pyplot as plt
import numpy as np
import simpy
from Framework import PropagationModel
from Framework.AirInterface import AirInterface
from Framework.EnergyProfile import EnergyProfile
from Framework.Gateway import Gateway
from Framework.Location import Location
from Framework.LoRaParameters import LoRaParameters
from Framework.Node import Node
from Framework.SNRModel import SNRModel
from ReplayMemory import ReplayMemory


def plot_environment(nodes, gateway):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(gateway.x, gateway.y, s=200, c='r', label="Gateway")

    x = []
    y = []

    for n in nodes:
        x.append(n.location.x)
        y.append(n.location.y)

    ax.scatter(x, y, s=50, c='b', label="End Node")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.legend()
    plt.show()


class LoraEnvironment():

    def __init__(self, policy_net, tau, seed, adr_rl=True, adr=True):
        super(LoraEnvironment, self).__init__()

        self.confirmed_messages = True
        self.printed = False
        #self.scaling_factor = 0.1
        #self.transmission_rate_bit_per_ms = self.scaling_factor * (12 * 8) / (60 * 60 * 1000)
        self.simulation_time = 1 * 60 * 60 * 1000 #/self.scaling_factor
        self.sigma = 7.8  # [0, 5, 7.8, 15, 20]
        self.MAX_DELAY_BEFORE_SLEEP_MS = 500
        self.MAX_DELAY_START_PER_NODE_MS = 30*60*1000

        self.env = simpy.Environment()

        self.replay_memory = ReplayMemory(1000)

        self.gateway = Gateway(self.env, Location(x=0, y=0, indoor=False), policy_net, self.replay_memory, tau, seed, adr_rl, adr, printed=self.printed)

        self.air_interface = AirInterface(self.gateway, PropagationModel.LogShadow(std=self.sigma), SNRModel(), self.env, self.printed)

        self.nodes = []

    def run(self):

        random.seed(1)

        tx_power_mW = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}
        rx_measurements = {'pre_mW': 8.2, 'pre_ms': 3.4, 'rx_lna_on_mW': 39,
                           'rx_lna_off_mW': 34, 'post_mW': 8.3, 'post_ms': 10.7}

        for node_id in range(100):
            energy_profile = EnergyProfile(5.7e-3, 15, tx_power_mW, rx_power=rx_measurements)

            lora_param = LoRaParameters(freq=np.random.choice(LoRaParameters.DEFAULT_CHANNELS),
                                        sf=7,
                                        bw=125,
                                        cr=5,
                                        crc_enabled=1,
                                        de_enabled=0,
                                        header_implicit_mode=0,
                                        tp=14)

            payload_size = 50 #random.randint(1, 100)

            node = Node(node_id,
                        energy_profile,
                        lora_param,
                        sleep_time=5*60*60*1000, # sleep 5 minute
                        process_time=1*60*60*1000, # process 10 seconds
                        adr=True,
                        location=Location(random.uniform(-2000, 2000), random.uniform(-2000, 2000)),
                        base_station=self.gateway,
                        env=self.env,
                        payload_size=payload_size,
                        air_interface=self.air_interface,
                        MAX_DELAY_START_PER_NODE_MS=self.MAX_DELAY_START_PER_NODE_MS,
                        MAX_DELAY_BEFORE_SLEEP_MS=self.MAX_DELAY_BEFORE_SLEEP_MS,
                        confirmed_messages=self.confirmed_messages,
                        printed=self.printed)

            self.nodes.append(node)

            self.env.process(node.run())

        #plot_environment(nodes, self.gateway.location)

        self.env.run()

        return self.gateway.get_score(), self.replay_memory

    def get_stats(self):

        mean_energy_per_bit_list = []

        for n in self.nodes:
            mean_energy_per_bit_list.append(n.energy_per_bit())

        return sum(mean_energy_per_bit_list)/len(mean_energy_per_bit_list)



