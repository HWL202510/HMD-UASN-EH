import numpy as np
import math


# The environment program can be written according to the literature, here we give an example.

class Env:
    def __init__(self):
        self.net_size = [500, 500]
        self.RELAY_NUM = 5
        self.MAX_VELOCITY = 1

        self.reward_weight = 1
        self.C_min = 0

        self.P_S = 1
        self.P_R = [1, 6]

        self.B = 10

        self.f = 10
        self.w = 8
        self.s = 1

        self.init_B = 8
        self.B_r = [self.init_B] * self.RELAY_NUM

        self.record_c = []
        self.c = 0
        self.c_all = 0

        self.interrupt_count = [0] * (self.RELAY_NUM + 1)
        self.record_interrupt_all = []

        self.history_battery = [[] for _ in range(self.RELAY_NUM)]
        self.history_G_sr = []
        self.history_G_rd = []
        self.history_G_sd = []

        self.rayleigh_gain_sr = None
        self.rayleigh_gain_rd = None
        self.rayleigh_gain_sd = None

    def harvest_energy_for_relays(self):
        harvested_energies = [np.random.uniform(0.0, 3.0) for _ in range(self.RELAY_NUM)]
        return harvested_energies

    def update_relay_energy_levels(self, selected_relay, actual_power_used):
        self.B_r[selected_relay] = max(0, self.B_r[selected_relay] - actual_power_used)

        new_energies = self.harvest_energy_for_relays()

        for relay_idx in range(self.RELAY_NUM):
            self.B_r[relay_idx] = min(self.init_B, self.B_r[relay_idx] + new_energies[relay_idx])
            self.history_battery[relay_idx].append(self.B_r[relay_idx])

        return self.B_r.copy()

    def generate_initial_relay_positions(self):
        grid_width, grid_length = self.net_size
        x_coords = np.random.random(self.RELAY_NUM) * grid_width
        y_coords = np.random.random(self.RELAY_NUM) * grid_length
        positions = np.vstack([x_coords, y_coords])
        return positions

    def compute_movement_parameters(self):
        movement_angles = np.random.random(self.RELAY_NUM) * 2 * np.pi
        movement_speeds = np.random.exponential(0.5, self.RELAY_NUM) * self.MAX_VELOCITY
        movement_speeds = np.clip(movement_speeds, 0, self.MAX_VELOCITY)
        return movement_angles, movement_speeds

    def calculate_new_positions(self, current_positions, angles, speeds):
        delta_x = speeds * np.cos(angles)
        delta_y = speeds * np.sin(angles)

        new_pos = current_positions.copy()
        new_pos[0, :] += delta_x
        new_pos[1, :] += delta_y

        width, height = self.net_size

        out_of_left = new_pos[0, :] <= 0
        out_of_right = new_pos[0, :] > width
        new_pos[0, out_of_left] = width
        new_pos[0, out_of_right] = 50

        out_of_bottom = new_pos[1, :] <= 0
        out_of_top = new_pos[1, :] > height
        new_pos[1, out_of_bottom] = height
        new_pos[1, out_of_top] = 50

        return new_pos

    def refresh_relay_positions(self, current_positions):
        move_angles, move_speeds = self.compute_movement_parameters()
        return self.calculate_new_positions(current_positions, move_angles, move_speeds)

    def calculate_source_to_relay_distances(self, relay_positions):
        distances = np.sqrt(np.square(relay_positions[0, :]) + np.square(relay_positions[1, :]))
        return distances

    def calculate_relay_to_destination_distances(self, relay_positions):
        dest_x, dest_y = self.net_size
        dx = dest_x - relay_positions[0, :]
        dy = dest_y - relay_positions[1, :]
        return np.sqrt(dx ** 2 + dy ** 2)

    def calculate_source_destination_distance(self):
        dest_x, dest_y = self.net_size
        return math.sqrt(dest_x ** 2 + dest_y ** 2)

    def generate_rayleigh_fading(self, scale_param, sample_size):
        return np.random.rayleigh(scale_param, sample_size)

    def generate_episode_channel_gains(self, scale_param, episode_length):
        self.rayleigh_gain_sr = self.generate_rayleigh_fading(scale_param, episode_length)
        self.rayleigh_gain_rd = self.generate_rayleigh_fading(scale_param, episode_length)
        self.rayleigh_gain_sd = self.generate_rayleigh_fading(scale_param, episode_length)

    def shadow_fading_effect(self, distance):
        alpha_coef = (0.11 * self.f ** 2 / (1 + self.f ** 2) +
                      44 * self.f ** 2 / (4100 + self.f ** 2) +
                      2.75e-4 * self.f ** 2 + 0.003) / 10
        alpha_val = 10 ** alpha_coef
        return (distance / 1000) ** -1.5 * alpha_val ** (-distance / 1000)

    def compute_environmental_noise(self, f=None, s=None, w=None):
        # 使用传入的参数，如果没有传入则使用实例属性
        freq = f if f is not None else self.f
        ship_param = s if s is not None else self.s
        wind_param = w if w is not None else self.w

        N_thermal = 17 - 30 * np.log10(freq)
        N_ship = 40 + 20 * (ship_param - 0.5) + 26 * np.log10(freq) - 60 * np.log10(freq + 0.03)
        N_wind = 50 + 7.5 * np.sqrt(wind_param) + 20 * np.log10(freq) - 40 * np.log10(freq + 0.4)
        N_turbulence = 20 * np.log10(freq) - 15

        total_noise = 10 ** (N_thermal / 10) + 10 ** (N_ship / 10) + 10 ** (N_wind / 10) + 10 ** (N_turbulence / 10)
        return 10 * np.log10(total_noise)

    def compute_channel_characteristics(self, time_slot):
        sr_gains = []
        rd_gains = []

        for i in range(self.RELAY_NUM):
            path_loss_sr = self.shadow_fading_effect(self.dis_sr[i])
            path_loss_rd = self.shadow_fading_effect(self.dis_rd[i])
            sr_gains.append(path_loss_sr * self.rayleigh_gain_sr[time_slot])
            rd_gains.append(path_loss_rd * self.rayleigh_gain_rd[time_slot])

        self.gain_sr = np.array(sr_gains)
        self.gain_rd = np.array(rd_gains)
        path_loss_sd = self.shadow_fading_effect(self.dis_sd)
        self.gain_sd = path_loss_sd * self.rayleigh_gain_sd[time_slot]

    def setup_initial_network_topology(self):
        self.relay_location = self.generate_initial_relay_positions()
        self.dis_sr = self.calculate_source_to_relay_distances(self.relay_location)
        self.dis_rd = self.calculate_relay_to_destination_distances(self.relay_location)
        self.dis_sd = self.calculate_source_destination_distance()

    def update_network_topology(self):
        self.relay_location = self.refresh_relay_positions(self.relay_location)
        self.dis_sr = self.calculate_source_to_relay_distances(self.relay_location)
        self.dis_rd = self.calculate_relay_to_destination_distances(self.relay_location)
        self.dis_sd = self.calculate_source_destination_distance()

    def calculate_signal_to_interference_ratio(self, transmit_power, channel_gain, noise_level):
        return (transmit_power * channel_gain) / noise_level

    def compute_capacity_metrics(self, relay_power, gain_sr, gain_rd, gain_sd):
        noise_power = self.compute_environmental_noise()

        sinr_sd = self.calculate_signal_to_interference_ratio(self.P_S, gain_sd, noise_power)
        sinr_sr = self.calculate_signal_to_interference_ratio(self.P_S, gain_sr, noise_power)
        sinr_rd = self.calculate_signal_to_interference_ratio(relay_power, gain_rd, noise_power)

        total_capacity = 0.5 * self.B * np.log2(1 + sinr_sd + (sinr_sr * sinr_rd) / (1 + sinr_sr + sinr_rd))

        return total_capacity

    def calculate_power_penalty(self, power_used, harvested_energy):
        return -5.2 / (1 + np.exp(-power_used + 5.0 * harvested_energy))

    def calculate_interruption_penalty(self, power_level):
        return -0.48 * power_level

    def get_environment_state(self, time_slot):
        battery_levels = [self.history_battery[i][time_slot] for i in range(self.RELAY_NUM)]
        return np.array(battery_levels)

    def execute_environment_step(self, action, time_slot):
        relay_selection = int(action[0])
        power_allocation = action[1]

        current_battery = self.B_r[relay_selection]
        current_gain_sr = self.gain_sr[relay_selection]
        current_gain_rd = self.gain_rd[relay_selection]
        current_gain_sd = self.gain_sd

        if current_battery < power_allocation:
            actual_power_used = 0
            harvested_energy = self.harvest_energy_for_relays()[relay_selection]
            penalty = self.calculate_power_penalty(power_allocation, harvested_energy)
            interruption_penalty = self.calculate_interruption_penalty(power_allocation)
            reward_value = penalty + interruption_penalty
            self.c_all = 0

            self.interrupt_count[relay_selection + 1] += 1
            self.interrupt_count[0] += 1
        else:
            actual_power_used = power_allocation
            total_capacity = self.compute_capacity_metrics(power_allocation, current_gain_sr,
                                                              current_gain_rd, current_gain_sd)
            harvested_energy = self.harvest_energy_for_relays()[relay_selection]
            penalty = self.calculate_power_penalty(power_allocation, harvested_energy)
            reward_value = penalty + (total_capacity - self.C_min)
            self.c_all = total_capacity
            self.c += total_capacity

        self.history_G_sr.append(current_gain_sr)
        self.history_G_rd.append(current_gain_rd)
        self.history_G_sd.append(current_gain_sd)

        for i in range(self.RELAY_NUM):
            self.history_battery[i].append(self.B_r[i])

        self.update_relay_energy_levels(relay_selection, actual_power_used)

        self.update_network_topology()
        self.compute_channel_characteristics(time_slot + 1)

        next_state = self.get_environment_state(time_slot)

        episode_complete = (time_slot == 179)
        if episode_complete:
            self.record_c.append(self.c)
            self.record_interrupt_all.append(self.interrupt_count[0])
            self.c = 0
            self.interrupt_count = [0] * (self.RELAY_NUM + 1)
            self.initialize_environment_state()

        return [next_state, reward_value, episode_complete, self.gain_sr, self.gain_rd, self.gain_sd]

    def initialize_environment_state(self):
        self.setup_initial_network_topology()
        self.generate_episode_channel_gains(10, 181)
        self.compute_channel_characteristics(0)

        self.B_r = [self.init_B] * self.RELAY_NUM

        self.history_battery = [[] for _ in range(self.RELAY_NUM)]
        self.history_G_sr = []
        self.history_G_rd = []
        self.history_G_sd = []

        for i in range(self.RELAY_NUM):
            self.history_battery[i].append(self.B_r[i])

        self.history_G_sr.append(self.gain_sr[0])
        self.history_G_rd.append(self.gain_rd[0])
        self.history_G_sd.append(self.gain_sd)

        initial_state = np.array([self.init_B] * self.RELAY_NUM)
        return initial_state, self.gain_sr, self.gain_rd, self.gain_sd, np.array(self.B_r)


Env.energy_harvesting = Env.harvest_energy_for_relays
Env.relay_energy_to_use = Env.update_relay_energy_levels
Env.get_init_relay_location = Env.generate_initial_relay_positions
Env.update_topology_everySecond = Env.refresh_relay_positions
Env.dis_s_r = Env.calculate_source_to_relay_distances
Env.dis_r_d = Env.calculate_relay_to_destination_distances
Env.dis_s_d = Env.calculate_source_destination_distance
Env.rayleigh_gain = Env.generate_rayleigh_fading
Env.all_rayleigh_gain_episode = Env.generate_episode_channel_gains
Env.shadow_fading = Env.shadow_fading_effect
Env.environment_noise = Env.compute_environmental_noise
Env.channel_gain = Env.compute_channel_characteristics
Env.create_init_topology = Env.setup_initial_network_topology
Env.brush_topology = Env.update_network_topology
Env.SINR_S_R = lambda self, P_s, G_sr, N: self.calculate_signal_to_interference_ratio(P_s, G_sr, N)
Env.SINR_S_D = lambda self, P_s, G_sd, N: self.calculate_signal_to_interference_ratio(P_s, G_sd, N)
Env.SINR_R_D = lambda self, P_r, G_rd, N: self.calculate_signal_to_interference_ratio(P_r, G_rd, N)
Env.C_relay_and_all = Env.compute_capacity_metrics
Env.panel = Env.calculate_power_penalty
Env.interrupt_reward = Env.calculate_interruption_penalty
Env.get_state = Env.get_environment_state
Env.step = Env.execute_environment_step
Env.initialize_state = Env.initialize_environment_state