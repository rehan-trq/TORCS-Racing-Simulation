import gym
from gym import spaces
import numpy as np
import snakeoil3_gym as snake
import os
import time
import copy
import collections as collections

class TorcsEnvironment:
    IDLE_LIMIT = 100
    SPEED_THRESHOLD = 5
    TARGET_SPEED = 50
    first_launch = True

    def __init__(self, vision_enabled=False, throttle_control=False, manual_gear=False):
        self.use_vision = vision_enabled
        self.enable_throttle = throttle_control
        self.allow_gear_change = manual_gear

        self.first_round = True

        os.system('pkill torcs')
        time.sleep(0.5)
        if self.use_vision:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

        if not self.enable_throttle:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if not self.use_vision:
            obs_high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            obs_low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
        else:
            obs_high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            obs_low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])

        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

    def reset(self, relaunch=False):
        self.current_step = 0

        if not self.first_launch:
            self.torcs_client.R.d['meta'] = True
            self.torcs_client.respond_to_server()

            if relaunch:
                self._restart_torcs()
                print(">>> TORCS relaunched")

        self.torcs_client = snake.Client(p=3001, vision=self.use_vision)
        self.torcs_client.MAX_STEPS = np.inf
        self.torcs_client.get_servers_input()
        observation_raw = self.torcs_client.S.d
        self.last_action = None
        self.first_launch = False
        self.latest_observation = self._format_observation(observation_raw)
        return self.get_observation()

    def step(self, agent_input):
        client = self.torcs_client
        torcs_input = self._translate_action(agent_input)
        control_packet = client.R.d

        control_packet['steer'] = torcs_input['steer']

        if not self.enable_throttle:
            if client.S.d['speedX'] < self.TARGET_SPEED - (control_packet['steer'] * 50):
                control_packet['accel'] += 0.01
            else:
                control_packet['accel'] -= 0.01
            control_packet['accel'] = min(control_packet['accel'], 0.2)

            if client.S.d['speedX'] < 10:
                control_packet['accel'] += 1.0 / (client.S.d['speedX'] + 0.1)

            slip = (client.S.d['wheelSpinVel'][2] + client.S.d['wheelSpinVel'][3]) - \
                   (client.S.d['wheelSpinVel'][0] + client.S.d['wheelSpinVel'][1])
            if slip > 5:
                control_packet['accel'] -= 0.2
        else:
            control_packet['accel'] = torcs_input.get('accel', 0)
            control_packet['brake'] = torcs_input.get('brake', 0)

        if self.allow_gear_change:
            control_packet['gear'] = torcs_input['gear']
        else:
            speed = client.S.d['speedX']
            control_packet['gear'] = 1 + (speed > 50) + (speed > 80) + (speed > 110) + \
                                     (speed > 140) + (speed > 170)

        previous_state = copy.deepcopy(client.S.d)

        client.respond_to_server()
        client.get_servers_input()

        updated_state = client.S.d
        self.latest_observation = self._format_observation(updated_state)

        reward = self._calculate_reward(previous_state, updated_state)
        done = self._check_termination(updated_state, previous_state)

        if done:
            self.first_round = False
            client.respond_to_server()

        self.current_step += 1

        return self.get_observation(), reward, client.R.d['meta'], {}

    def end(self):
        os.system('pkill torcs')

    def get_observation(self):
        return self.latest_observation

    def _restart_torcs(self):
        os.system('pkill torcs')
        time.sleep(0.5)
        command = 'torcs -nofuel -nodamage -nolaptime -vision &' if self.use_vision else 'torcs -nofuel -nolaptime &'
        os.system(command)
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

    def _translate_action(self, input_array):
        action = {'steer': input_array[0]}
        if self.enable_throttle:
            action['accel'] = input_array[1]
            action['brake'] = input_array[2]
        if self.allow_gear_change:
            action['gear'] = int(input_array[3])
        return action

    def _calculate_reward(self, prev_obs, current_obs):
        velocity = np.array(current_obs['speedX'])
        angle = current_obs['angle']
        track_position = current_obs['trackPos']
        reward_progress = velocity * np.cos(angle) - abs(velocity * np.sin(angle)) - velocity * abs(track_position)

        if current_obs['damage'] > prev_obs['damage']:
            return -1
        return reward_progress

    def _check_termination(self, obs, prev_obs):
        if np.cos(obs['angle']) < 0:
            self.torcs_client.R.d['meta'] = True
            return True
        return False

    def convert_vision_data(self, flat_image_array):
        r = flat_image_array[::3]
        g = flat_image_array[1::3]
        b = flat_image_array[2::3]
        r, g, b = np.reshape(r, (64, 64)), np.reshape(g, (64, 64)), np.reshape(b, (64, 64))
        return np.array([r, g, b], dtype=np.uint8)

    def _format_observation(self, raw_data):
        if not self.use_vision:
            keys = ['focus', 'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                    'opponents', 'rpm', 'track', 'trackPos', 'wheelSpinVel']
            FormattedObs = collections.namedtuple("FormattedObs", keys)

            return FormattedObs(
                focus=np.array(raw_data['focus']) / 200.0,
                speedX=raw_data['speedX'] / 300.0,
                speedY=raw_data['speedY'] / 300.0,
                speedZ=raw_data['speedZ'] / 300.0,
                angle=raw_data['angle'],
                damage=raw_data['damage'],
                opponents=np.array(raw_data['opponents']) / 200.0,
                rpm=raw_data['rpm'],
                track=np.array(raw_data['track']) / 200.0,
                trackPos=raw_data['trackPos'],
                wheelSpinVel=np.array(raw_data['wheelSpinVel']) / 100.0
            )
        else:
            # You can extend this to vision observations similarly
            pass
