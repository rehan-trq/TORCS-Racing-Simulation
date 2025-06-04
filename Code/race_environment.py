import gym
from gym import spaces
import numpy as np
import gym_torcs.snakeoil3_gym as snakeoil3
import copy
import collections as col
import os
import subprocess
import psutil
import time
import math
import random
from xml.etree import ElementTree as ET

FLOAT32_TYPE = np.float32

class RaceEnvironment(gym.Env):
    speed_check_delay = 150.0  # Further relaxed for recovery
    min_progress_speed = 3.0  # Adjusted for recovery attempts
    base_speed = 300.0

    def __init__(self, vision_mode=False, use_throttle=False, enable_gear_shift=False,
                 track_config_file=None, track_speed=1.0, enable_render=True,
                 enable_damage=False, max_laps=1, record_data=False,
                 add_noise=False, max_record_episodes=1, max_record_steps=3600,
                 record_id=0, relaunch_interval=11, randomize_track=False,
                 track_reuse_limit=500, instance_rank=0):
        """Initialize the TORCS racing environment."""
        if track_config_file is None:
            track_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "track_configs/standard.xml")

        if use_throttle and enable_gear_shift:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1]),
                high=np.array([1.0, 1.0, 6]),
                dtype=[FLOAT32_TYPE, FLOAT32_TYPE, np.int32]
            )
        elif use_throttle:
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=FLOAT32_TYPE
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-1.0]),
                high=np.array([1.0]),
                dtype=FLOAT32_TYPE
            )

        self.use_vision = vision_mode
        self.use_throttle = use_throttle
        self.use_gear_shift = enable_gear_shift
        self.track_speed = track_speed
        self.render_enabled = enable_render
        self.damage_enabled = enable_damage
        self.record_enabled = record_data
        self.noise_enabled = add_noise
        self.randomize_enabled = randomize_track
        self.track_reuse_count = 0
        self.reuse_limit = track_reuse_limit
        self.first_run = True
        self.process_id = None
        self.track_config = track_config_file
        self.restart_count = 1
        self.relaunch_interval = relaunch_interval
        self.max_laps = max_laps
        self.max_record_episodes = max_record_episodes
        self.max_record_steps = max_record_steps
        self.record_id = record_id

        launch_args = ["torcs", "-nofuel", "-nolaptime", "-a", str(self.track_speed)]
        if not self.damage_enabled:
            launch_args.append("-nodamage")
        if self.noise_enabled:
            launch_args.append("-noisy")
        if self.use_vision:
            launch_args.append("-vision")
        if not self.render_enabled:
            launch_args.append("-T")
        if self.track_config:
            launch_args.extend(["-raceconfig", self.track_config])
        if self.record_enabled:
            launch_args.extend([
                f"-rechum {self.record_id}",
                f"-recepisodelim {self.max_record_episodes}",
                f"-rectimesteplim {self.max_record_steps}"
            ])
        launch_args.append("&")
        self.process_id = subprocess.Popen(launch_args, shell=False).pid

        self.random_seed = 42

        if not self.use_vision:
            high_bounds = np.hstack((
                math.pi,
                np.ones(19),
                np.array([np.inf] * 4),
                np.array([500.0] * 4),
                np.inf,
                np.ones(36),
                1.0  # Track curvature
            ))
            low_bounds = np.hstack((
                -math.pi,
                np.full(19, -1.0),
                np.array([-np.inf] * 4),
                np.zeros(4),
                0.0,
                np.full(36, -1.0),
                0.0  # Track curvature
            ))
            self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=FLOAT32_TYPE)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)

        if self.randomize_enabled:
            self.generate_track_config()

    def generate_track_config(self):
        """Create a randomized track configuration."""
        if self.track_reuse_count == 0 or self.track_reuse_count % self.reuse_limit == 0:
            track_length = 2700
            max_position = int(0.7 * track_length)
            agent_start = random.randint(0, 20) * 10
            opponent_count = random.randint(1, 10)
            min_position = agent_start + 50
            max_step = math.floor((max_position - min_position) / opponent_count / 100) * 100
            opponent_positions = [random.randint(min_position, min_position + max_step) for _ in range(opponent_count)]
            min_position += max_step

            config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track_configs/random")
            os.makedirs(config_dir, exist_ok=True)
            config_filename = f"track_{agent_start}_{'_'.join(map(str, opponent_positions))}.xml"
            config_path = os.path.join(config_dir, config_filename)

            if not os.path.isfile(config_path):
                xml_tree = ET.ElementTree()
                race_root = ET.Element("race")
                drivers = race_root.find(".//section[@name='Drivers']") or ET.SubElement(race_root, "section", name="Drivers")
                drivers.append(ET.Element("attnum", name="max_drivers", val=str(1 + opponent_count)))
                drivers.append(ET.Element("attstr", name="main_module", val="scr_server"))
                drivers.append(ET.Element("attnum", name="main_idx", val="1"))

                agent_node = ET.SubElement(drivers, "section", name="1")
                agent_node.append(ET.Element("attnum", name="index", val="0"))
                agent_node.append(ET.Element("attstr", name="module", val="scr_server"))
                drivers.append(ET.Element("attnum", name="start_pos_1", val=str(agent_start)))

                for idx, pos in enumerate(opponent_positions):
                    bot_node = ET.SubElement(drivers, "section", name=str(2 + idx))
                    bot_node.append(ET.Element("attnum", name="index", val=str(2 + idx)))
                    bot_node.append(ET.Element("attstr", name="module", val="fixed"))
                    drivers.append(ET.Element("attnum", name=f"start_pos_{idx + 1}", val=str(pos)))

                xml_tree._setroot(race_root)
                xml_tree.write(config_path)

            self.track_config = config_path
            self.track_reuse_count = 1

    def set_seed(self, seed_value=42):
        """Set random seed."""
        self.random_seed = seed_value

    def advance(self, action):
        """Advance the simulation with the given action."""
        torcs_interface = self.torcs_interface
        torcs_action = self.map_action_to_torcs(action)
        control_data = torcs_interface.R.controls

        control_data['steer'] = torcs_action['steer']
        if self.use_throttle:
            control_data['accel'] = torcs_action['accel']
            control_data['brake'] = torcs_action['brake']

        # Gear logic with reverse for stuck situations
        is_stuck = (torcs_interface.S.state_data['speedX'] < 5 and min(torcs_interface.S.state_data['track']) < 0) or min(torcs_interface.S.state_data['opponents']) < 10
        if is_stuck:
            control_data['gear'] = -1  # Reverse gear
        elif self.use_gear_shift:
            control_data['gear'] = torcs_action['gear']
        else:
            control_data['gear'] = 1
            speed = torcs_interface.S.state_data['speedX']
            rpm = torcs_interface.S.state_data['rpm']
            if speed > 50 and rpm > 7000:
                control_data['gear'] = 2
            if speed > 80 and rpm > 7000:
                control_data['gear'] = 3
            if speed > 110 and rpm > 7000:
                control_data['gear'] = 4
            if speed > 140 and rpm > 7000:
                control_data['gear'] = 5
            if speed > 170 and rpm > 7000:
                control_data['gear'] = 6
            if rpm < 3500 and control_data['gear'] > 1:
                control_data['gear'] -= 1

        print(f"Advance: steer={control_data['steer']:.3f}, accel={control_data.get('accel', 0):.3f}, brake={control_data.get('brake', 0):.3f}, gear={control_data['gear']}")

        prev_state = copy.deepcopy(torcs_interface.S.state_data)
        torcs_interface.send_controls()
        torcs_interface.fetch_server_data()
        current_state = torcs_interface.S.state_data
        self.state_data = self.process_state(current_state)

        track_data = np.array(current_state['track'])
        track_position = np.array(current_state['trackPos'])
        speed_x = np.array(current_state['speedX'])
        damage = np.array(current_state['damage'])
        rpm = np.array(current_state['rpm'])
        opponents = np.array(current_state['opponents'])
        progress = speed_x * np.cos(current_state['angle'])
        reward = progress

        # Enhanced reward shaping
        if track_data.min() < 0:
            reward -= 20  # Stronger off-track penalty
        if abs(track_position) > 1:
            reward -= 5 * abs(track_position)  # Penalty for straying
        if min(opponents) < 10:  # Collision penalty
            reward -= 10
        if np.cos(current_state['angle']) > 0.9:  # Reward track alignment
            reward += 5 * np.cos(current_state['angle'])
        if control_data['gear'] == -1 and speed_x < 0:  # Reward reversing when stuck
            reward += 10
        if control_data['brake'] > 0.5 and track_data.min() >= 0:
            reward -= 2  # Penalize excessive braking on track

        terminate = False
        if current_state['damage'] - prev_state['damage'] > 0:
            reward = -10
            terminate = True
            control_data['meta'] = True

        if track_data.min() < 0:
            terminate = True
            control_data['meta'] = True

        if self.step_count > self.speed_check_delay and progress < self.min_progress_speed:
            terminate = True
            control_data['meta'] = True

        if np.cos(current_state['angle']) < 0:
            terminate = True
            control_data['meta'] = True

        if int(current_state["lap"]) > self.max_laps:
            terminate = True
            control_data['meta'] = True

        if control_data['meta']:
            self.first_run = False
            torcs_interface.send_controls()

        self.step_count += 1
        print(f"State: speedX={speed_x:.2f}, trackPos={track_position:.2f}, rpm={rpm:.0f}, opponent_min={min(opponents):.2f}, reward={reward:.2f}, terminated={terminate}")

        return self.get_state(), reward, control_data['meta'], {}

    def restart(self, relaunch=False):
        """Reset the environment for a new episode."""
        self.step_count = 0
        if not self.initial_restart:
            self.torcs_interface.R.controls['meta'] = True
            self.torcs_interface.send_controls()

            if relaunch or self.restart_count % self.relaunch_interval == 0:
                self.relaunch_server()
                self.restart_count = 1
                print("### TORCS Server Relaunched ###")

        if self.randomize_enabled:
            self.generate_track_config()

        self.torcs_interface = snakeoil3.Client(
            p=3001, vision=self.use_vision, process_id=self.process_id,
            race_config_path=self.track_config, race_speed=self.track_speed,
            rendering=self.render_enabled, lap_limiter=self.max_laps,
            damage=self.damage_enabled, recdata=self.record_enabled,
            noisy=self.noise_enabled, rec_index=self.record_id,
            rec_episode_limit=self.max_record_episodes,
            rec_timestep_limit=self.max_record_steps
        )
        self.torcs_interface.max_steps = np.inf
        torcs_interface = self.torcs_interface
        torcs_interface.fetch_server_data()
        state = torcs_interface.S.state_data
        self.state_data = self.process_state(state)
        self.last_action = None
        self.initial_restart = False
        self.process_id = torcs_interface.process_id
        self.restart_count += 1
        self.track_reuse_count += 1
        return self.get_state()

    def shutdown(self):
        """Terminate the TORCS server process."""
        if self.process_id:
            try:
                proc = psutil.Process(self.process_id)
                for child in proc.children():
                    child.terminate()
                proc.terminate()
            except Exception:
                self.process_id = None

    def close(self):
        """Close the environment."""
        self.shutdown()

    def get_state(self):
        """Return the current observation."""
        return self.state_data

    def relaunch_server(self):
        """Relaunch the TORCS server process."""
        if self.process_id:
            try:
                proc = psutil.Process(self.process_id)
                for child in proc.children():
                    child.terminate()
                proc.terminate()
            except Exception:
                pass

        if self.randomize_enabled:
            self.generate_track_config()

        launch_args = ["torcs", "-nofuel", "-nolaptime", "-a", str(self.track_speed)]
        if not self.damage_enabled:
            launch_args.append("-nodamage")
        if self.noise_enabled:
            launch_args.append("-noisy")
        if self.use_vision:
            launch_args.append("-vision")
        if not self.render_enabled:
            launch_args.append("-T")
        if self.track_config:
            launch_args.extend(["-raceconfig", self.track_config])
        if self.record_enabled:
            launch_args.extend([
                f"-rechum {self.record_id}",
                f"-recepisodelim {self.max_record_episodes}",
                f"-rectimesteplim {self.max_record_steps}"
            ])
        launch_args.append("&")
        self.process_id = subprocess.Popen(launch_args, shell=False).pid

    def map_action_to_torcs(self, action):
        """Convert agent action to TORCS format."""
        torcs_action = {'steer': action[0]}
        if self.use_throttle:
            torcs_action.update({'accel': action[1], 'brake': action[2]})
        if self.use_gear_shift:
            torcs_action.update({'gear': action[3]})
        return torcs_action

    def convert_vision_data(self, image_vector):
        """Transform vision data into RGB image."""
        pixels = []
        for i in range(0, 12286, 3):
            pixel = [image_vector[i], image_vector[i + 1], image_vector[i + 2]]
            pixels.append(pixel)
        return np.array(pixels, dtype=np.uint8)

    def process_state(self, raw_state):
        """Convert raw server state to structured observation."""
        if not self.use_vision:
            fields = ['focus', 'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                      'opponents', 'rpm', 'track', 'trackPos', 'wheelSpinVel', 'lap', 'curvature']
            State = col.namedtuple('State', fields)
            track = np.array(raw_state['track'])
            curvature = np.std(track[:9] - track[10:]) / 200.0  # Normalized curvature
            return State(
                focus=np.array(raw_state['focus'], dtype=FLOAT32_TYPE) / 200.0,
                speedX=np.array(raw_state['speedX'], dtype=FLOAT32_TYPE) / 300.0,
                speedY=np.array(raw_state['speedY'], dtype=FLOAT32_TYPE) / 300.0,
                speedZ=np.array(raw_state['speedZ'], dtype=FLOAT32_TYPE) / 300.0,
                angle=np.array(raw_state['angle'], dtype=FLOAT32_TYPE) / math.pi,
                damage=np.array(raw_state['damage'], dtype=FLOAT32_TYPE),
                opponents=np.array(raw_state['opponents'], dtype=FLOAT32_TYPE) / 200.0,
                rpm=np.array(raw_state['rpm'], dtype=FLOAT32_TYPE) / 10000.0,
                track=np.array(raw_state['track'], dtype=FLOAT32_TYPE) / 200.0,
                trackPos=np.array(raw_state['trackPos'], dtype=FLOAT32_TYPE),
                wheelSpinVel=np.array(raw_state['wheelSpinVel'], dtype=FLOAT32_TYPE) / 100.0,
                lap=np.array(raw_state['lap'], dtype=np.uint8),
                curvature=np.array(curvature, dtype=FLOAT32_TYPE)
            )
        else:
            fields = ['focus', 'speedX', 'speedY', 'speedZ', 'opponents', 'rpm',
                      'track', 'wheelSpinVel', 'img', 'curvature']
            State = col.namedtuple('State', fields)
            rgb_image = self.convert_vision_data(raw_state['img'])
            track = np.array(raw_state['track'])
            curvature = np.std(track[:9] - track[10:]) / 200.0
            return State(
                focus=np.array(raw_state['focus'], dtype=FLOAT32_TYPE) / 200.0,
                speedX=np.array(raw_state['speedX'], dtype=FLOAT32_TYPE) / self.base_speed,
                speedY=np.array(raw_state['speedY'], dtype=FLOAT32_TYPE) / self.base_speed,
                speedZ=np.array(raw_state['speedZ'], dtype=FLOAT32_TYPE) / self.base_speed,
                opponents=np.array(raw_state['opponents'], dtype=FLOAT32_TYPE) / 200.0,
                rpm=np.array(raw_state['rpm'], dtype=FLOAT32_TYPE),
                track=np.array(raw_state['track'], dtype=FLOAT32_TYPE) / 200.0,
                wheelSpinVel=np.array(raw_state['wheelSpinVel'], dtype=FLOAT32_TYPE),
                img=rgb_image,
                curvature=np.array(curvature, dtype=FLOAT32_TYPE)
            )